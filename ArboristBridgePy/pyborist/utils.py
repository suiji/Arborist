from os import listdir, path
import shutil


def move_dynamic_files(src_dir='.', dst_dir='pyborist'):
    """This is a special helper function.

    This function moves the compiled dynamic libraries in the src_dir to dst_dir.
    """
    exts = ['.pyd', '.so', '.dynlib']
    for f in listdir(src_dir):
        for ext in exts:
            if f.endswith(ext):
                shutil.move(path.join(src_dir, f), path.join(dst_dir, f))
    print('dynamic files moved')
    return True
