
try:
    import cPickle as pickle
except:
    import pickle


def write_pickle_file(obj, write_file, write_file_prefix=None, close_after_write=True):

    if isinstance(write_file, file):
        assert not write_file.closed
        pickle.dump(obj, write_file)

        if close_after_write:
            write_file.close()

        return True

    elif isinstance(write_file, (str, unicode)):
        # TODO: output_file includes filename and path
        with open(write_file, "wb") as wfile:
            pickle.dump(obj, wfile)

        return True

    else:

        return False


def read_pickle_file(read_file, close_after_read=True):

    read_results = None

    if isinstance(read_file, file):
        assert not read_file.closed
        read_results = pickle.load(read_file)

        if close_after_read:
            read_file.close()

    elif isinstance(read_file, (str, unicode)):
        # TODO: output_file includes filename and path
        with open(read_file, "rb") as rfile:
            read_results = pickle.load(rfile)

    return read_results


if __name__ == '__main__':
    pass




