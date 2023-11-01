import argparse
import os


class ValidateModelPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            # sys.exit("This path must be exists" % values)
            raise Exception("This path must be exists" % values)
        else:
            abspath = os.path.abspath(values)
            setattr(namespace, self.dest, abspath)


class ValidatePathExists(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            # sys.exit("This path must be exists" % values)
            raise Exception("This path must be exists" % values)
        else:
            abspath = os.path.abspath(values)
            setattr(namespace, self.dest, abspath)
