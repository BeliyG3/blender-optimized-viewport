# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Locating the DLSS feature libraries the NVIDIA driver has already installed.

The build does not ship `nvngx_dlssd.dll` or `nvngx_dlssg.dll`. NVIDIA's licence for the DLSS SDK
does not permit redistributing them alongside a GPL application, and the DLSS pull request NVIDIA
itself opened against Blender ships them no more than this does.

They do not have to be shipped, because the driver already installs them. Under the NGX model root
sit per-feature, per-version directories holding a file named `160_<id>.bin`, and that file is the
feature library: an ordinary PE, signed by NVIDIA, carrying `nvngx_dlssd.dll` in its own export
table. What it is not is *findable* - NGX resolves feature libraries by the literal name
`nvngx_dlssd.dll` along its search paths, while the `.bin` cache is a separate lookup keyed by an
application id that never matches Blender's. So the file has to be copied under the name NGX looks
for. That is all this module does, and all the button in the preferences does.
"""

from __future__ import annotations

import sys


# NGX feature directories, and the file name each one has to be installed as.
FEATURES = {
    "dlssd": ("nvngx_dlssd.dll", "DLSS Ray Reconstruction"),
    "dlssg": ("nvngx_dlssg.dll", "DLSS Frame Generation"),
}


def is_supported():
    """Whether this platform can have driver-installed feature libraries at all."""
    return sys.platform == "win32"


def ngx_models_root():
    """The directory the driver installs feature libraries under, or None.

    Read from the same registry key the renderer trusts to find `_nvngx.dll` itself, rather than
    assuming `C:\\ProgramData\\NVIDIA\\NGX`: an installation that put NGX elsewhere would otherwise
    look to this module like a machine with no driver at all.
    """
    if not is_supported():
        return None

    import os
    import winreg

    for key_path in (
        r"System\CurrentControlSet\Services\nvlddmkm\Parameters\NGXCore",
        r"System\CurrentControlSet\Services\nvlddmkm\NGXCore",
    ):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                ngx_path, _ = winreg.QueryValueEx(key, "NGXPath")
        except OSError:
            continue

        if not isinstance(ngx_path, str) or not os.path.isabs(ngx_path):
            continue

        # `NGXPath` points at the directory holding `_nvngx.dll`, and the models sit beside that
        # directory rather than inside it - so the models are looked for next to it first, and
        # under ProgramData only as the layout the installer uses by default.
        #
        # `dirname` rather than trimming separators by hand: on "C:\\" the hand-written version
        # left "C:", and joining onto a bare drive letter gives a path relative to that drive's
        # current directory rather than to its root.
        parent = os.path.dirname(os.path.normpath(ngx_path))
        candidates = [os.path.join(ngx_path, "models")]
        if parent:
            candidates.insert(0, os.path.join(parent, "models"))
        program_data = os.environ.get("ProgramData", "")
        if os.path.isabs(program_data):
            candidates.append(os.path.join(program_data, "NVIDIA", "NGX", "models"))

        for models in candidates:
            # Checked again at the end: a candidate built from a drive root can still come out
            # drive-relative, and returning one would point the loader at the wrong place.
            if os.path.isabs(models) and os.path.isdir(models):
                return models

    return None


def _sort_key_from_directory(name):
    """`20317696` -> `(310, 6, 0)`, for ordering only.

    The directory name packs a version, but it is the version of the driver's model package, not of
    every file inside it: under `20316673` the Ray Reconstruction library reports 310.2.0 while the
    directory says 310.2.1. Good enough to sort by, not good enough to show.
    """
    try:
        packed = int(name)
    except ValueError:
        return None
    if packed <= 0:
        return None
    return ((packed >> 16) & 0xFFFF, (packed >> 8) & 0xFF, packed & 0xFF)


def _file_version(path):
    """The version a PE reports about itself, as `"310.6.0"`, or None.

    Asked of the file rather than inferred from its directory, so that what the button offers is
    what NGX will later report having loaded.
    """
    import ctypes
    from ctypes import wintypes

    version = ctypes.WinDLL("version.dll")

    size = version.GetFileVersionInfoSizeW(ctypes.c_wchar_p(path), None)
    if not size:
        return None

    buffer = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(ctypes.c_wchar_p(path), 0, size, buffer):
        return None

    block = ctypes.c_void_p()
    length = wintypes.UINT()
    if not version.VerQueryValueW(
            buffer, ctypes.c_wchar_p("\\"), ctypes.byref(block), ctypes.byref(length)):
        return None

    class VS_FIXEDFILEINFO(ctypes.Structure):
        _fields_ = [
            ("dwSignature", wintypes.DWORD),
            ("dwStrucVersion", wintypes.DWORD),
            ("dwFileVersionMS", wintypes.DWORD),
            ("dwFileVersionLS", wintypes.DWORD),
            ("dwProductVersionMS", wintypes.DWORD),
            ("dwProductVersionLS", wintypes.DWORD),
            ("dwFileFlagsMask", wintypes.DWORD),
            ("dwFileFlags", wintypes.DWORD),
            ("dwFileOS", wintypes.DWORD),
            ("dwFileType", wintypes.DWORD),
            ("dwFileSubtype", wintypes.DWORD),
            ("dwFileDateMS", wintypes.DWORD),
            ("dwFileDateLS", wintypes.DWORD),
        ]

    info = ctypes.cast(block, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
    if info.dwSignature != 0xFEEF04BD:
        return None

    return "{:d}.{:d}.{:d}".format(
        info.dwFileVersionMS >> 16,
        info.dwFileVersionMS & 0xFFFF,
        info.dwFileVersionLS >> 16,
    )


def available_versions(feature):
    """Feature libraries the driver has installed, newest first.

    Each entry is `(version_tuple, version_text, path)`. Signatures are not checked here - listing
    is drawn on every redraw of the preferences, and verifying a 60 MB file is not something to do
    at that rate. The check happens once, in `install`, before anything is copied.
    """
    root = ngx_models_root()
    if root is None:
        return []

    import os

    versions_dir = os.path.join(root, feature, "versions")
    if not os.path.isdir(versions_dir):
        return []

    found = []
    try:
        entries = os.listdir(versions_dir)
    except OSError:
        return []

    for entry in entries:
        sort_key = _sort_key_from_directory(entry)
        if sort_key is None:
            continue
        files_dir = os.path.join(versions_dir, entry, "files")
        if not os.path.isdir(files_dir):
            continue
        try:
            candidates = [f for f in os.listdir(files_dir) if f.lower().endswith(".bin")]
        except OSError:
            continue
        for candidate in candidates:
            path = os.path.join(files_dir, candidate)
            version_text = _file_version(path)
            if version_text is None:
                # Not a versioned PE, so not one of the libraries being looked for.
                continue
            found.append((sort_key, version_text, path))

    found.sort(key=lambda item: item[0], reverse=True)
    return found


# Enum items for the versions the driver provides, held between calls.
#
# Blender does not copy the strings a dynamic enum callback returns, it keeps pointing at them, so a
# list built fresh on every call leaves the UI holding freed strings. Kept per feature, because the
# two libraries need not be present at the same versions, and handed back unchanged when the driver
# is still offering the same set, so that a redraw does not swap the strings out from under an open
# menu.
_version_enum_items = {}


def version_enum_items(feature):
    """Items for a dynamic `EnumProperty` over the versions this driver provides, newest first."""
    label = FEATURES[feature][1] if feature in FEATURES else feature

    items = [
        (version_text, version_text, "Install {:s} {:s}".format(label, version_text))
        for _sort_key, version_text, _path in available_versions(feature)
    ]
    if not items:
        # An empty enum can be neither drawn nor executed, so there is always one item; the UI does
        # not offer the list at all in this case.
        items = [('NONE', "None", "This driver provides no {:s} library".format(label))]

    cached = _version_enum_items.get(feature)
    if cached == items:
        return cached
    _version_enum_items[feature] = items
    return items


def target_directory():
    """Where feature libraries are installed to.

    The user's own data directory rather than the directory holding `blender.exe`: a Blender
    unpacked into Program Files is not writable, and asking the user to run as administrator to
    enable a denoiser is a poor trade. The renderer adds this directory to the NGX search path.
    """
    import bpy

    return bpy.utils.user_resource('DATAFILES', path="dlss", create=True)


def version_directory(version_text):
    """Where one installed version lives.

    A directory per version, rather than one file overwritten in place. NGX loads a feature library
    as soon as anything asks whether the feature is supported - which the preferences do, on every
    redraw - and Windows will not let a loaded library be written over. Installing into a directory
    that does not exist yet cannot collide with a file that is already open.
    """
    import os

    return os.path.join(target_directory(), version_text)


def _pointer_path(feature):
    """The file naming the version to load. Written here, read by the renderer."""
    import os

    return os.path.join(target_directory(), "{:s}.active".format(feature))


def _sort_key_from_version_text(text):
    """`"310.7.129"` -> `(310, 7, 129)`, or None for anything that is not a version."""
    parts = text.split(".")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def installed_versions(feature):
    """Versions installed under the data directory, newest first, as `(sort_key, text, path)`."""
    import os

    file_name = FEATURES[feature][0]

    found = []
    try:
        entries = os.listdir(target_directory())
    except OSError:
        return found

    for entry in entries:
        sort_key = _sort_key_from_version_text(entry)
        if sort_key is None:
            continue
        path = os.path.join(target_directory(), entry, file_name)
        if os.path.isfile(path):
            found.append((sort_key, entry, path))

    found.sort(key=lambda item: item[0], reverse=True)
    return found


def active_version(feature):
    """The version the renderer loads on its next start, or None.

    None means there is nothing to point at, and the renderer falls back to the flat layout - which
    is where a build older than this one left its library, and where it goes on working untouched.
    """
    import os

    try:
        with open(_pointer_path(feature), "r", encoding="utf-8") as pointer:
            version_text = pointer.read().strip()
    except OSError:
        return None

    if not version_text:
        return None

    file_name = FEATURES[feature][0]
    if not os.path.isfile(os.path.join(version_directory(version_text), file_name)):
        # Pointed at a version that has since been removed by hand.
        return None

    return version_text


# What was active when this session started, per feature. An install changes what the pointer says;
# it cannot change what NGX already holds open, and the preferences have to be able to say both.
_active_at_startup = {}


def active_version_at_startup(feature):
    """The version this session actually loaded.

    Recorded the first time it is asked and never revised. `install` asks before it writes, so the
    answer is the state that preceded any install done from here.
    """
    if feature not in _active_at_startup:
        _active_at_startup[feature] = active_version(feature)
    return _active_at_startup[feature]


def installed_path(feature):
    """The library the renderer will load, or None.

    Looked for the way the renderer looks: the version the pointer names, then the newest installed
    one, then the flat layout an older add-on wrote, and finally the executable directory, for a
    library placed beside `blender.exe` by hand.
    """
    import os
    import bpy

    file_name = FEATURES[feature][0]

    version_text = active_version(feature)
    if version_text is not None:
        return os.path.join(version_directory(version_text), file_name)

    versions = installed_versions(feature)
    if versions:
        return versions[0][2]

    for directory in (target_directory(), os.path.dirname(bpy.app.binary_path)):
        candidate = os.path.join(directory, file_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def installed_version(feature):
    """What is installed for a feature, as `(path, version_text)`.

    Both are None when nothing is installed; `version_text` alone is None for a file that reports no
    version, which a hand-placed library may well do. Asked of the installed file rather than
    remembered from the install, so that a library put there by other means is described correctly.
    """
    path = installed_path(feature)
    if path is None:
        return None, None
    return path, _file_version(path)


# Who the libraries this module installs have to be signed by, read from the organisation (O)
# field of the signing certificate. The simple display name is not used: it falls back through
# common name, organisation and e-mail depending on what the certificate carries, so a match
# against it does not say which field matched.
EXPECTED_SIGNER = "NVIDIA Corporation"


def _signer_name(path):
    """The display name on the certificate the file was signed with, or None.

    A valid signature only says that somebody the machine trusts signed the file. Which publisher
    it was is a separate question, and it is the one that matters here: this module copies the file
    into the place the renderer loads from, so a validly signed library from anyone at all would
    otherwise be installed as though NVIDIA had shipped it.
    """
    import ctypes
    from ctypes import wintypes

    crypt32 = ctypes.WinDLL("crypt32.dll")

    CERT_QUERY_OBJECT_FILE = 1
    CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED = 1 << 10
    CERT_QUERY_FORMAT_FLAG_BINARY = 1 << 1
    CMSG_SIGNER_CERT_INFO_PARAM = 7
    ENCODING = 0x00000001 | 0x00010000  # X509_ASN_ENCODING | PKCS_7_ASN_ENCODING
    CERT_FIND_SUBJECT_CERT = 11 << 16
    CERT_NAME_ATTR_TYPE = 3
    ORGANIZATION_OID = b"2.5.4.10"  # szOID_ORGANIZATION_NAME

    crypt32.CertFindCertificateInStore.restype = ctypes.c_void_p
    crypt32.CertGetNameStringW.restype = wintypes.DWORD

    store = wintypes.HANDLE()
    message = wintypes.HANDLE()
    if not crypt32.CryptQueryObject(CERT_QUERY_OBJECT_FILE,
                                    ctypes.c_wchar_p(path),
                                    CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
                                    CERT_QUERY_FORMAT_FLAG_BINARY,
                                    0, None, None, None,
                                    ctypes.byref(store), ctypes.byref(message), None):
        return None

    certificate = None
    try:
        size = wintypes.DWORD()
        if not crypt32.CryptMsgGetParam(message, CMSG_SIGNER_CERT_INFO_PARAM, 0, None,
                                        ctypes.byref(size)):
            return None
        signer_info = ctypes.create_string_buffer(size.value)
        if not crypt32.CryptMsgGetParam(message, CMSG_SIGNER_CERT_INFO_PARAM, 0, signer_info,
                                        ctypes.byref(size)):
            return None

        certificate = crypt32.CertFindCertificateInStore(store, ENCODING, 0, CERT_FIND_SUBJECT_CERT,
                                                         signer_info, None)
        if not certificate:
            return None

        organisation = ctypes.c_char_p(ORGANIZATION_OID)
        length = crypt32.CertGetNameStringW(ctypes.c_void_p(certificate), CERT_NAME_ATTR_TYPE, 0,
                                            organisation, None, 0)
        if length <= 1:
            return None
        name = ctypes.create_unicode_buffer(length)
        crypt32.CertGetNameStringW(ctypes.c_void_p(certificate), CERT_NAME_ATTR_TYPE, 0,
                                   organisation, name, length)
        return name.value
    finally:
        if certificate:
            crypt32.CertFreeCertificateContext(ctypes.c_void_p(certificate))
        if message:
            crypt32.CryptMsgClose(message)
        if store:
            crypt32.CertCloseStore(store, 0)


def _has_valid_signature(path):
    """Whether the file carries a valid Authenticode signature from the expected publisher.

    Checked because this module copies an executable out of a directory it does not own and puts it
    somewhere the renderer will load it from. Both halves matter: that the signature verifies at
    all, and that the organisation on the certificate is NVIDIA's. A file that fails either has
    been replaced by something, and loading it would be the renderer's problem rather than this
    module's.

    Call this on the copy that will be installed, never on the file it was copied from: a check
    against the source says nothing about the bytes that end up in place, because the source can
    change between the two.
    """
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class WINTRUST_FILE_INFO(ctypes.Structure):
        _fields_ = [
            ("cbStruct", wintypes.DWORD),
            ("pcwszFilePath", wintypes.LPCWSTR),
            ("hFile", wintypes.HANDLE),
            ("pgKnownSubject", ctypes.POINTER(GUID)),
        ]

    class WINTRUST_DATA(ctypes.Structure):
        _fields_ = [
            ("cbStruct", wintypes.DWORD),
            ("pPolicyCallbackData", ctypes.c_void_p),
            ("pSIPClientData", ctypes.c_void_p),
            ("dwUIChoice", wintypes.DWORD),
            ("fdwRevocationChecks", wintypes.DWORD),
            ("dwUnionChoice", wintypes.DWORD),
            ("pFile", ctypes.POINTER(WINTRUST_FILE_INFO)),
            ("dwStateAction", wintypes.DWORD),
            ("hWVTStateData", wintypes.HANDLE),
            ("pwszURLReference", wintypes.LPWSTR),
            ("dwProvFlags", wintypes.DWORD),
            ("dwUIContext", wintypes.DWORD),
            ("pSignatureSettings", ctypes.c_void_p),
        ]

    # WINTRUST_ACTION_GENERIC_VERIFY_V2
    action = GUID(
        0x00AAC56B, 0xCD44, 0x11D0,
        (ctypes.c_ubyte * 8)(0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE),
    )

    file_info = WINTRUST_FILE_INFO()
    file_info.cbStruct = ctypes.sizeof(WINTRUST_FILE_INFO)
    file_info.pcwszFilePath = path
    file_info.hFile = None
    file_info.pgKnownSubject = None

    trust_data = WINTRUST_DATA()
    trust_data.cbStruct = ctypes.sizeof(WINTRUST_DATA)
    trust_data.dwUIChoice = 2          # WTD_UI_NONE
    trust_data.fdwRevocationChecks = 0  # WTD_REVOKE_NONE
    trust_data.dwUnionChoice = 1        # WTD_CHOICE_FILE
    trust_data.pFile = ctypes.pointer(file_info)
    trust_data.dwStateAction = 1        # WTD_STATEACTION_VERIFY
    trust_data.dwProvFlags = 0x00000010  # WTD_SAFER_FLAG

    wintrust = ctypes.WinDLL("wintrust.dll")
    result = wintrust.WinVerifyTrust(None, ctypes.byref(action), ctypes.byref(trust_data))

    # The verification allocates state that has to be handed back whatever the answer was.
    trust_data.dwStateAction = 2        # WTD_STATEACTION_CLOSE
    wintrust.WinVerifyTrust(None, ctypes.byref(action), ctypes.byref(trust_data))

    if result != 0:
        return False

    signer = _signer_name(path)
    return signer is not None and signer.strip().lower() == EXPECTED_SIGNER.lower()


def install(feature, version_text=""):
    """Copy a driver-installed feature library into place under the name NGX looks for.

    Returns `(True, message)` or `(False, message)`; the caller reports it. An empty `version_text`
    takes the newest available.
    """
    import os
    import shutil

    if feature not in FEATURES:
        return False, "Unknown DLSS feature: {:s}".format(feature)

    file_name, label = FEATURES[feature]

    candidates = available_versions(feature)
    if not candidates:
        return False, (
            "No {:s} library found in the NVIDIA driver. Update the driver, or place {:s} "
            "next to blender.exe manually".format(label, file_name)
        )

    if version_text:
        candidates = [c for c in candidates if c[1] == version_text]
        if not candidates:
            return False, "{:s} version {:s} is not installed by the driver".format(
                label, version_text)

    _, chosen_text, source = candidates[0]

    refused = (
        "The {:s} library is not signed by {:s}, or its signature does not verify, and it was "
        "not installed".format(label, EXPECTED_SIGNER)
    )

    # Asked before anything is written, so that the preferences can go on reporting what this
    # session loaded rather than what it will load next time.
    active_version_at_startup(feature)

    destination_directory = version_directory(chosen_text)
    destination = os.path.join(destination_directory, file_name)

    if os.path.isfile(destination):
        # Already there, from an earlier install of the same version. Copying over it would fail
        # if this is the version NGX is holding open, and there is nothing to copy anyway - but it
        # is still what the renderer will load, so it is checked rather than trusted for having
        # been installed once.
        if not _has_valid_signature(destination):
            return False, refused
    else:
        # The copy is made first and checked afterwards, on the copy itself. Checking the driver's
        # file instead would say nothing about what ends up here: it is opened again to be copied,
        # and whatever answers on that second open is what gets installed. Verifying the copy binds
        # the answer to those bytes.
        #
        # The staging file is created exclusively, under a name nothing else will pick: a fixed
        # name would be shared with any other Blender installing the same version at the same
        # moment, and then one process could verify what the other had just written. It lives in
        # the destination directory so that putting it in place is a rename rather than a second
        # copy, and it is removed on every way out of here.
        import tempfile

        try:
            os.makedirs(destination_directory, exist_ok=True)
            handle, incoming = tempfile.mkstemp(dir=destination_directory,
                                                prefix=file_name + ".", suffix=".incoming")
            os.close(handle)
        except OSError as error:
            return False, "Could not install {:s}: {:s}".format(file_name, str(error))

        installed = False
        try:
            try:
                shutil.copy2(source, incoming)
            except OSError as error:
                return False, "Could not install {:s}: {:s}".format(file_name, str(error))

            if not _has_valid_signature(incoming):
                return False, refused

            try:
                os.replace(incoming, destination)
            except OSError as error:
                return False, "Could not install {:s}: {:s}".format(file_name, str(error))
            installed = True
        finally:
            # Reached on the way out however this block ends, including an exception raised by the
            # signature check itself. After a successful rename there is nothing left to remove.
            if not installed:
                try:
                    os.remove(incoming)
                except OSError:
                    pass

    try:
        with open(_pointer_path(feature), "w", encoding="utf-8") as pointer:
            pointer.write(chosen_text)
    except OSError as error:
        return False, "Copied {:s} but could not record it as the one to load: {:s}".format(
            file_name, str(error))

    if active_version_at_startup(feature) == chosen_text:
        return True, "{:s} {:s} is already the version in use".format(label, chosen_text)

    return True, "{:s} {:s} installed. Restart Blender to use it".format(label, chosen_text)


def remove(feature, version_text):
    """Delete one installed version.

    The version this session loaded cannot be deleted: NGX holds the file open for as long as the
    process lives. That is reported as what it is, rather than as a copy that went wrong.
    """
    import os

    if feature not in FEATURES:
        return False, "Unknown DLSS feature: {:s}".format(feature)

    file_name, label = FEATURES[feature]
    directory = version_directory(version_text)
    path = os.path.join(directory, file_name)

    if not os.path.isfile(path):
        return False, "{:s} {:s} is not installed".format(label, version_text)

    try:
        os.remove(path)
    except OSError as error:
        if getattr(error, "winerror", None) == 32:
            return False, (
                "{:s} {:s} is the version this session is using. Restart Blender, then remove "
                "it".format(label, version_text)
            )
        return False, "Could not remove {:s} {:s}: {:s}".format(label, version_text, str(error))

    # The directory is shared between the two features when both were installed at the same version,
    # so it goes only once nothing is left in it.
    try:
        os.rmdir(directory)
    except OSError:
        pass

    if active_version(feature) is None:
        # The pointer named what was just removed. Fall back to the newest of what is left, so that
        # the next start has something to load rather than nothing.
        remaining = installed_versions(feature)
        try:
            if remaining:
                with open(_pointer_path(feature), "w", encoding="utf-8") as pointer:
                    pointer.write(remaining[0][1])
            elif os.path.isfile(_pointer_path(feature)):
                os.remove(_pointer_path(feature))
        except OSError:
            pass

    return True, "{:s} {:s} removed".format(label, version_text)
