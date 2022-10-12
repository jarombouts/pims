"""Reader for Norpix .seq files

Author: Nathan C. Keim
Based heavily on cine.py by Kleckner and Caswell
"""
from PIL import Image
import io

from pims.frame import Frame
from pims.base_frames import FramesSequence, index_attr
from pims.utils.misc import FileLocker
import os, struct, itertools
from warnings import warn
import datetime
import numpy as np
from threading import Lock

__all__ = [
    "NorpixSeq",
]

DWORD = "L"  # 4 bytes
UINT = "I"  # 4 bytes
LONG = "l"  # 4 bytes
LONGLONG = "q"  # 8 bytes
DOUBLE = "d"  # 8 bytes
USHORT = "H"  # 2 bytes
SHORT = "h"  # 2 bytes

HEADER_FIELDS = [
    ("magic", DWORD),
    ("name", "24s"),
    ("version", LONG),
    ("header_size", LONG),
    ("description", "512s"),
    ("width", DWORD),
    ("height", DWORD),
    ("bit_depth", DWORD),
    ("bit_depth_real", DWORD),
    ("image_size_bytes", DWORD),
    ("image_format", DWORD),
    ("allocated_frames", DWORD),
    ("origin", DWORD),
    ("true_image_size", DWORD),
    ("suggested_frame_rate", DOUBLE),
    ("description_format", LONG),
    ("reference_frame", DWORD),
    ("fixed_size", DWORD),
    ("flags", DWORD),
    ("bayer_pattern", LONG),
    ("time_offset_us", LONG),
    ("extended_header_size", LONG),
    ("compression_format", DWORD),  # 'eHCompression enum', compression format used
    ("reference_time_s", LONG),
    ("reference_time_ms", USHORT),
    ("reference_time_us", USHORT),
    ("h264_gop", UINT),  # 'ulong', Group of Picture value, if h264 is used
    ("h264_bitrate", UINT),  # 'ulong', bitrate value, if h264 is used
    # 'ulong', "JPEG Format Quality and lossless" => split into two shorts
    ("jpeg_quality", USHORT),
    ("jpeg_lossless", UINT),
    ("h264_decode_format", DWORD),  # 'eHImageFormat enum', H264 decode format
    ("index_offset", LONGLONG),  # 'long long', offset of compression index data
    # 'long', oldest_frame_index will be > 0 if loop recording into circular buffer
    ("oldest_frame_index", LONG),
    ("bytes_alignment", LONG)  # 'long', image alignment (for uncompressed sequences)
    # More header values not implemented
]

# Enum for possible compression formats in a .SEQ file
EHCOMPRESSION = {
    0: "H_COMPRESSION_NONE",
    1: "H_COMPRESSION_JPEG",
    2: "H_COMPRESSION_RLE",  # Not implemented yet!
    3: "H_COMPRESSION_HUFFMAN",  # Not implemented yet!
    4: "H_COMPRESSION_LZ",  # Not implemented yet!
    5: "H_COMPRESSION_RLE_FAST",  # Not implemented yet!
    6: "H_COMPRESSION_HUFFMAN_FAST",  # Not implemented yet!
    7: "H_COMPRESSION_LZ_FAST",  # Not implemented yet!
    8: "H_COMPRESSION_H264",  # Not implemented yet!
    9: "H_COMPRESSION_WAVELET",  # Not implemented yet!
}

# Enum for possible image formats
EHIMAGEFORMAT = {
    0: "H_IMAGE_UNKNOWN",
    100: "H_IMAGE_MONO",
    101: "H_IMAGE_MONO_BAYER",
    200: "H_IMAGE_BGR",
    300: "H_IMAGE_PLANAR",
    400: "H_IMAGE_RGB",
    500: "H_IMAGE_BGRx",
    600: "H_IMAGE_YUV422",
    610: "H_IMAGE_YUV422_20",
    700: "H_IMAGE_UVY422",
    800: "H_IMAGE_UVY411",
    900: "H_IMAGE_UVY444",
    905: "H_IMAGE_BGR555_PACKED",
    906: "H_IMAGE_BGR565_PACKED",
    112: "H_IMAGE_MONO_MSB",
    113: "H_IMAGE_MONO_BAYER_MSB",
    114: "H_IMAGE_MONO_MSB_SWAP",
    115: "H_IMAGE_MONO_BAYER_MSB_SWAP",
    123: "H_IMAGE_BGR10_PPACKED",
    124: "H_IMAGE_BGR10_PACKED_PHOENIX",
    125: "H_IMAGE_RGB10_PACKED_PHOENIX",
    131: "H_IMAGE_MONO_PACKED",
    132: "H_IMAGE_MONO_BAYER_PACKED",
    133: "H_IMAGE_MONO_PACKED_8448",
    134: "H_IMAGE_MONO_BAYER_PACKED_8448",
    135: "H_IMAGE_GVSP_BGR10V1_PACKED",
    136: "H_IMAGE_GVSP_BGR10V2_PACKED",
    1000: "H_IMAGE_BASER_VENDOR_SPECIFIC",
    1001: "H_IMAGE_EURESYS_JPEG",
    1002: "H_IMAGE_ISG_JPEG",
}


class NorpixSeq(FramesSequence):
    """Read Norpix sequence (.seq) files

    This is the native format of StreamPix software, owned by NorPix Inc.
    The format is described in the StreamPix documentation.

    This reader only supports uncompressed files (with the same number
    of bytes for each frame).

    Only unsigned 8, 16 or 32 bit monochrome files are directly supported.
    Other color and monochrome pixel formats can be handled by setting
    the as_raw flag which results in frames with unsigned 8-bit data type.
    and the correct number of rows.  (If a raw frame size in bytes is not
    evenly divisible by the number of rows, a 1-D array is returned).

    Nominally thread-safe.

    Parameters
    ----------
    filename : string
        Path to the .seq file
    as_raw : boolean, optional
        Required for non-monochrome frames.
        Images will be returned as an ndarray of bytes.
        2-dimensional if the image height evenly divides the bytes per image,
        1-dimensional otherwise.
    """

    @classmethod
    def class_exts(cls):
        return {"seq"} | super(NorpixSeq, cls).class_exts()

    propagate_attrs = [
        "frame_shape",
        "pixel_type",
        "get_time",
        "get_time_float",
        "filename",
        "width",
        "height",
        "frame_rate",
    ]

    def __init__(self, filename, as_raw=False):
        super(NorpixSeq, self).__init__()
        self._file = open(filename, "rb")
        self._filename = filename

        self.header_dict = self._read_header(HEADER_FIELDS)

        if self.header_dict["magic"] != 0xFEED:
            raise IOError("The format of this .seq file is unrecognized")
        if (
            EHCOMPRESSION[self.header_dict["compression_format"]]
            != "H_COMPRESSION_NONE"
        ):
            raise IOError("This reader only supports uncompressed frames")
        if self.header_dict["image_format"] != 100 and not as_raw:
            raise IOError(
                "Non-monochrome images are only supported as_raw in .seq files"
            )

        # File-level metadata
        if self.header_dict["version"] >= 5:  # StreamPix version 6
            self._image_offset = 8192
            # Timestamp = 4-byte unsigned long + 2-byte unsigned short (ms)
            #   + 2-byte unsigned short (us)
            self._timestamp_struct = struct.Struct("<LHH")
            self._timestamp_micro = True
        else:  # Older versions
            self._image_offset = 1024
            self._timestamp_struct = struct.Struct("<LH")
            self._timestamp_micro = False
        self._image_block_size = self.header_dict["true_image_size"]
        self._filesize = os.stat(self._filename).st_size
        self._image_count = int(
            (self._filesize - self._image_offset) / self._image_block_size
        )

        # Image metadata
        self._width = self.header_dict["width"]
        self._height = self.header_dict["height"]
        self._image_bytes = self.header_dict["image_size_bytes"]
        if as_raw:
            self._pixel_count = self._image_bytes
            if self._pixel_count % self._height == 0:
                self._shape = (self._height, int(self._pixel_count / self._height))
            else:
                self._shape = (self._image_bytes,)
            self._dtype = "uint8"
        else:
            try:
                self._pixel_count = self._width * self._height
                dtype_native = "uint%i" % self.header_dict["bit_depth"]
                self._dtype = np.dtype(dtype_native)
                self._shape = (self._height, self._width)
            except TypeError as e:
                raise IOError(
                    self._dtype + " pixels not supported; use as_raw and convert"
                )

        # Public metadata
        self.metadata = {
            k: self.header_dict[k]
            for k in (
                "description",
                "bit_depth_real",
                "origin",
                "suggested_frame_rate",
                "width",
                "height",
            )
        }
        self.metadata["gamut"] = 2 ** self.metadata["bit_depth_real"] - 1

        self._file_lock = Lock()

    def _read_header(self, fields, offset=0):
        self._file.seek(offset)
        tmp = dict()
        for name, format in fields:
            val = self._unpack(format)
            tmp[name] = val

        return tmp

    def _unpack(self, fs, offset=None):
        if offset is not None:
            self._file.seek(offset)
        s = struct.Struct("<" + fs)
        vals = s.unpack(self._file.read(s.size))
        if len(vals) == 1:
            return vals[0]
        else:
            return vals

    def _verify_frame_no(self, i):
        if int(i) != i:
            raise ValueError("Frame numbers can only be integers")
        if i >= self._image_count or i < 0:
            raise ValueError("Frame number is out of range: " + str(i))

    def set_process_func(self, process_func):
        # Expose the _validate_process_func for use after the header is
        # available
        self._validate_process_func(process_func)

    def get_frame(self, i):
        self._verify_frame_no(i)
        with FileLocker(self._file_lock):
            self._file.seek(self._image_offset + self._image_block_size * i)
            imdata = np.fromfile(
                self._file, self.pixel_type, self._pixel_count
            ).reshape(self._shape)
            # Timestamp immediately follows
            tfloat, ts = self._read_timestamp()
            md = {"time": ts, "time_float": tfloat, "gamut": self.metadata["gamut"]}
            return Frame(imdata, frame_no=i, metadata=md)

    def _read_timestamp(self):
        """Read a timestamp at the current position in the file.

        Returns a floating-point representation in seconds, and a datetime instance.
        """
        if self._timestamp_micro:
            tsecs, tms, tus = self._timestamp_struct.unpack(self._file.read(8))
            tfloat = tsecs + float(tms) / 1000.0 + float(tus) / 1.0e6
        else:
            tsecs, tms = self._timestamp_struct.unpack(self._file.read(6))
            tfloat = tsecs + float(tms) / 1000.0
        return tfloat, datetime.datetime.fromtimestamp(tfloat)

    def _get_time(self, i):
        """Call _read_timestamp() for a given frame."""
        self._verify_frame_no(i)
        with FileLocker(self._file_lock):
            self._file.seek(
                self._image_offset + self._image_block_size * i + self._image_bytes
            )
            return self._read_timestamp()

    @index_attr
    def get_time(self, i):
        """Return the time of frame i as a datetime instance.

        Calling this function in a different timezone than where the movie
        was recorded will result in an offset. The .seq format does not
        store UTC or timezone information.
        """
        return self._get_time(i)[1]

    @index_attr
    def get_time_float(self, i):
        """Return the time of frame i as a floating-point number of seconds."""
        return self._get_time(i)[0]

    def dump_times_float(self):
        """Return all frame times in file, as an array of floating-point numbers."""
        return np.array([self._get_time(i)[0] for i in range(len(self))])

    @property
    def filename(self):
        return self._filename

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def width(self):
        return self.metadata["width"]

    @property
    def height(self):
        return self.metadata["height"]

    @property
    def frame_shape(self):
        return (self.metadata["height"], self.metadata["width"])

    @property
    def frame_rate(self):
        return self.metadata["suggested_frame_rate"]

    def __len__(self):
        return self._image_count

    def close(self):
        self._file.close()

    def __repr__(self):
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w}w x {h}h
Pixel Datatype: {dtype}""".format(
            filename=self.filename,
            count=len(self),
            h=self.frame_shape[0],
            w=self.frame_shape[1],
            dtype=self.pixel_type,
        )


class NorpixJPEGSeq(FramesSequence):
    """Read JPEG-compressed Norpix sequence (.seq) files

    This is the native format of StreamPix software, owned by NorPix Inc.
    The format is described in the StreamPix documentation.

    This reader supports JPEG-compressed files.

    Only unsigned 8, 16 or 32 bit monochrome files are directly supported.
    Other color and monochrome pixel formats can be handled by setting
    the as_raw flag which results in frames with unsigned 8-bit data type.
    and the correct number of rows.  (If a raw frame size in bytes is not
    evenly divisible by the number of rows, a 1-D array is returned).

    Nominally thread-safe.

    Parameters
    ----------
    filename : string
        Path to the .seq file
    as_raw : boolean, optional
        Required for non-monochrome frames.
        Images will be returned as an ndarray of bytes.
        2-dimensional if the image height evenly divides the bytes per image,
        1-dimensional otherwise.
    """

    @classmethod
    def class_exts(cls):
        return {"seq"} | super(NorpixJPEGSeq, cls).class_exts()

    propagate_attrs = [
        "frame_shape",
        "pixel_type",
        "get_time",
        "get_time_float",
        "filename",
        "width",
        "height",
        "frame_rate",
    ]

    def __init__(self, filename):
        super(NorpixJPEGSeq, self).__init__()
        self._file = open(filename, "rb")
        self._filename = filename

        self.header_dict = self._read_header(HEADER_FIELDS)

        if self.header_dict["magic"] != 0xFEED:
            raise IOError("The format of this .seq file is unrecognized")
        if (
            EHCOMPRESSION[self.header_dict["compression_format"]]
            != "H_COMPRESSION_JPEG"
        ):
            raise IOError("This reader only supports JPEG compressed frames")
        if self.header_dict["image_format"] not in EHIMAGEFORMAT.keys():
            raise IOError(
                f"Detected image format {self.header_dict['image_format']}; "
                f"this is not supported"
            )

        # File-level metadata
        if self.header_dict["version"] >= 5:  # StreamPix version 6
            self._image_offset = 1024  # Frame data is stored on aligned 8k boundaries => NO APPEARS 1024 FOR JPEG?!
            # Timestamp = 4-byte unsigned long + 2-byte unsigned short (ms)
            #   + 2-byte unsigned short (us)
            self._timestamp_struct = struct.Struct("<LHH")
            self._timestamp_micro = True
        else:  # Older versions
            self._image_offset = 1024
            self._timestamp_struct = struct.Struct("<LH")
            self._timestamp_micro = False
        self._image_block_size = self.header_dict["true_image_size"]
        self._filesize = os.stat(self._filename).st_size

        self._sequence_index = self._prepare_sequence_index()
        self._image_count = len(self._sequence_index)

        # Image metadata
        self._width = self.header_dict["width"]
        self._height = self.header_dict["height"]
        # NOTE: this is the size in bytes of the uncompressed image!
        self._image_bytes = self.header_dict["image_size_bytes"]

        # todo adapt to reading JPEG data; can't simply infer from raw byte counts etc.
        try:
            self._pixel_count = self._width * self._height
            dtype_native = "uint%i" % self.header_dict["bit_depth_real"]
            self._dtype = np.dtype(dtype_native)
            self._shape = (
                self._height,
                self._width,
                self._image_bytes // self._pixel_count,
            )
        except TypeError as e:
            raise NotImplementedError(
                self._dtype + " bit depth pixels not supported yet"
            )

        # Public metadata
        self.metadata = {
            k: self.header_dict[k]
            for k in (
                "description",
                "bit_depth_real",
                "origin",
                "suggested_frame_rate",
                "width",
                "height",
                "jpeg_quality",
            )
        }
        self.metadata["gamut"] = 2 ** self.metadata["bit_depth_real"] - 1

        self._file_lock = Lock()

    def _prepare_sequence_index(self):
        """Read the .idx file, if it exists, and return a list of frame offsets"""
        idx_filename = self._filename + ".idx"
        if os.path.exists(idx_filename):
            self._idx_file = open(idx_filename, "rb")

            # a .idx file contains 20-byte chunks for each frame in the .seq file
            # LOL NO IT DOESN'T?!
            # turns out the chunks are 24 bytes long...
            fields = (
                ("offset", LONGLONG),  # 8 bytes
                ("image_size", LONG),  # 4 bytes = sums to 12
                ("timestamp", LONG),  # 4 bytes = sums to 16
                ("timestamp_ms", SHORT),  # 2 bytes = sums to 18
                ("timestamp_us", SHORT),  # 2 bytes = sums to 20
                ("whatever_this_is", LONG),  # 4 bytes = sums to 24
            )

            # seek to 0, read & unpack consecutive 20-byte chunks until end of file
            self._idx_file.seek(0)
            return [
                chunk
                for chunk in self._yield_idx_chunks(file=self._idx_file, fields=fields)
            ]
        # if it doesn't exist, we'll have to read the whole file and reconstruct it.
        else:
            return [chunk for chunk in self._reconstruct_sequence_index()]

    def _reconstruct_sequence_index(self):
        """
        If there's no .idx file, reconstruct its contents by scanning the .seq file

        JPEG data blocks are 4 bytes "JPEG data size + 4", the actual JPEG data,
        and then 8 bytes with the timestamp

        Will return a dict with the offset where a new data block starts,
        the size of the JPEG data block, and the timestamp
        """
        filesize = os.stat(self._filename).st_size

        # seek _file to 0 + header size
        self._file.seek(self._image_offset)
        # now iteratively read the first 4 bytes to get compressed image data size + 4

        while self._file.tell() < filesize:
            to_return = {}
            # get offset where this data block starts
            to_return["offset"] = self._file.tell()

            # read 4 bytes to get compressed image data size + 4; store actual image size
            image_size = (
                struct.unpack("<L", self._file.read(4))[0] - 4
            )  # SHREK IS LOVE, SHREK IS LIFE
            to_return["image_size"] = image_size

            # seek to timestamp and read
            _discard = self._file.read(image_size)
            to_return["timestamp"] = struct.unpack("<" + LONG, self._file.read(4))[0]
            to_return["timestamp_ms"] = struct.unpack("<" + SHORT, self._file.read(2))[
                0
            ]
            to_return["timestamp_us"] = struct.unpack("<" + SHORT, self._file.read(2))[
                0
            ]

            yield to_return

    @staticmethod
    def _yield_idx_chunks(file, fields):
        filesize = os.fstat(file.fileno()).st_size
        while file.tell() < filesize:
            to_return = {}
            for field_name, field_type in fields:
                s = struct.Struct("<" + field_type)
                values = s.unpack(file.read(s.size))
                if len(values) == 1:
                    to_return[field_name] = values[0]
                else:
                    to_return[field_name] = values

            # idx file stores JPEG data size + 4 for some reason;
            # clean up to make it useful for reading data
            to_return["image_size"] -= 4
            yield to_return

    # reads header fields from file
    def _read_header(self, fields, offset=0):
        self._file.seek(offset)
        tmp = dict()
        for name, format in fields:
            val = self._unpack(format)
            tmp[name] = val

        return tmp

    # used in unpacking header structs
    def _unpack(self, fs, offset=None):
        if offset is not None:
            self._file.seek(offset)
        s = struct.Struct("<" + fs)
        vals = s.unpack(self._file.read(s.size))
        if len(vals) == 1:
            return vals[0]
        else:
            return vals

    # used when getting time or frame data
    def _verify_frame_no(self, i):
        if int(i) != i:
            raise ValueError("Frame numbers can only be integers")
        if i >= self._image_count or i < 0:
            raise ValueError("Frame number is out of range: " + str(i))

    def set_process_func(self, process_func):
        # Expose the _validate_process_func for use after the header is
        # available
        self._validate_process_func(process_func)

    # this method should be present when inheriting from FramesSequence
    # TODO ADAPT TO READING JPEGS
    def get_frame(self, i):
        self._verify_frame_no(i)
        with FileLocker(self._file_lock):
            # seek to the offset of the frame we want
            # "offset" is the start of the data block;
            # + 4 to skip the first 4 bytes containing the size of the JPEG data
            jpeg_data_offset = self._sequence_index[i]["offset"] + 4
            self._file.seek(jpeg_data_offset)

            # read the JPEG data bytes with opencv
            jpeg_data_buffer = io.BytesIO(
                self._file.read(self._sequence_index[i]["image_size"])
            )
            jpeg_data_buffer.seek(0)

            # get buffer length and check if this is according to the frame index
            buffer_length = jpeg_data_buffer.getbuffer().nbytes
            assert buffer_length == self._sequence_index[i]["image_size"], (
                f"Unexpected number of bytes read from file: "
                f"expected {self._sequence_index[i]['image_size']} bytes, "
                f"got {buffer_length} bytes"
            )

            # open with Pillow and convert to numpy array
            pil_image = Image.open(jpeg_data_buffer)
            imdata = np.array(pil_image)

            # get timestamp as float and as python datetime object from index
            tfloat, ts = self._read_timestamp(i)

            # prepare metadata
            md = {"time": ts, "time_float": tfloat, "gamut": self.metadata["gamut"]}

            return Frame(imdata, frame_no=i, metadata=md)

    def _read_timestamp(self, i):
        """Read a timestamp at the current position in the file.

        Returns a floating-point representation in seconds, and a datetime instance.
        """
        tsecs = self._sequence_index[i]["timestamp"]
        tms = self._sequence_index[i]["timestamp_ms"]
        tus = self._sequence_index[i]["timestamp_us"]

        tfloat = tsecs + float(tms) / 1000.0 + float(tus) / 1.0e6

        return tfloat, datetime.datetime.fromtimestamp(tfloat)

    def _get_time(self, i):
        """Call _read_timestamp() for a given frame."""
        self._verify_frame_no(i)
        with FileLocker(self._file_lock):
            self._file.seek(
                self._image_offset + self._image_block_size * i + self._image_bytes
            )
            return self._read_timestamp()

    @index_attr
    def get_time(self, i):
        """Return the time of frame i as a datetime instance.

        Calling this function in a different timezone than where the movie
        was recorded will result in an offset. The .seq format does not
        store UTC or timezone information.
        """
        return self._get_time(i)[1]

    @index_attr
    def get_time_float(self, i):
        """Return the time of frame i as a floating-point number of seconds."""
        return self._get_time(i)[0]

    def dump_times_float(self):
        """Return all frame times in file, as an array of floating-point numbers."""
        return np.array([self._get_time(i)[0] for i in range(len(self))])

    @property
    def filename(self):
        return self._filename

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def width(self):
        return self.metadata["width"]

    @property
    def height(self):
        return self.metadata["height"]

    @property
    def frame_shape(self):
        return (self.metadata["height"], self.metadata["width"])

    @property
    def frame_rate(self):
        return self.metadata["suggested_frame_rate"]

    def __len__(self):
        return self._image_count

    def close(self):
        self._file.close()

    def __repr__(self):
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w}w x {h}h
Pixel Datatype: {dtype}""".format(
            filename=self.filename,
            count=len(self),
            h=self.frame_shape[0],
            w=self.frame_shape[1],
            dtype=self.pixel_type,
        )
