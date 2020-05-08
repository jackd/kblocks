# PR underway: https://github.com/google/gin-config/pull/25


import contextlib
import functools
import os

from gin.config import _FILE_READERS


@contextlib.contextmanager
def _change_dir_context(directory: str):
    """Change directory for the duration of the context block."""
    orig_dir = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(orig_dir)


@contextlib.contextmanager
def _open_from_directory(path: str):
    """Change directory to the containing folder while the file remains open.

  Allows for relative includes. See `enable_relative_includes`.
  """
    path = os.path.realpath(path)
    folder = os.path.dirname(path)
    with _change_dir_context(folder):
        with open(path, "r") as fp:
            yield fp


def _expand(path: str):
    return os.path.expanduser(os.path.expandvars(path))


def _exists_with_expand(path):
    """Check if the expanded path exists according to registered file readers."""
    path = _expand(path)
    for _, exists in _FILE_READERS:
        if exists is not _exists_with_expand and exists(path):
            return True
    return False


def _open_with_expand(path: str):
    """Open the expanded path with first suitable other registered file reader."""
    path = _expand(path)
    for reader, exists in _FILE_READERS:
        if reader is not _open_with_expand and exists(path):
            return reader(path)


def enable_variable_expansion(highest_priority: bool = True):
    """Allow variables and user symbols (~) in included files.

    Args:
      highest_priority: if True, expanded paths will be attempted first.
      See `register_file_reader`.
    """
    register_file_reader(
        _open_with_expand, _exists_with_expand, highest_priority=highest_priority
    )


def enable_relative_includes(highest_priority: bool = True):
    """Allow includes to be relative to the directory containing the config file.

    Args:
      highest_priority: if True, relative includes will take preferences over
        default behaviour.
    """
    register_file_reader(
        _open_from_directory, os.path.isfile, highest_priority=highest_priority
    )


def register_file_reader(*args, highest_priority: bool = False):
    """Register a file reader for use in parse_config_file.

    Registered file readers will be used to try reading files passed to
    `parse_config_file`. All file readers (beginning with the default `open`)
    will be tried until one of them succeeds at opening the file.

    This function may also be be used used as a decorator. For example:

        @register_file_reader(IOError)
        def exotic_data_source(filename):
          ...

    Args:
      *args: (When used as a decorator, only the existence check is supplied.)
        - file_reader_fn: The file reader function to register. This should be a
          function that can be used as a context manager to open a file and
          provide a file-like object, similar to Python's built-in `open`.
        - is_readable_fn: A function taking the file path and returning a boolean
          indicating whether the file can be read by `file_reader_fn`.
        - highest_priority: if True, this reader is tried before readers already
            registered. Note if other readers are subsequently registered with
            `highest_priority=True`, they will supersede earlier ones. Calling this
            with an already registered reader and highest_priority=True will move
            it to highest priority.

    Returns:
      `None`, or when used as a decorator, a function that will perform the
      registration using the supplied readability predicate.
    """

    def do_registration(file_reader_fn, is_readable_fn):
        try:
            index = [fr[0] for fr in _FILE_READERS].index(file_reader_fn)
            if highest_priority:
                del _FILE_READERS[index]
                # we'll add it back later
            else:
                # already present and not highest priority
                return file_reader_fn
        except ValueError:
            # not present
            pass

        # definitely not present now
        element = (file_reader_fn, is_readable_fn)
        if highest_priority:
            _FILE_READERS.insert(0, element)
        else:
            _FILE_READERS.append(element)
        return file_reader_fn

    if len(args) == 1:  # It's a decorator.
        return functools.partial(do_registration, is_readable_fn=args[0])
    elif len(args) == 2:
        do_registration(*args)
    else:  # 0 or > 2 arguments supplied.
        err_str = "register_file_reader() takes 1 or 2 arguments ({} given)"
        raise TypeError(err_str.format(len(args)))
