import argparse
from aml_io.log_utils import aml_logging

def main():
    parser = argparse.ArgumentParser(description='Train and test forward model')
    parser.add_argument('-l', '--logtype', type=str, help='-l debug -l info, -l warning, -l critical, -l error')
    args = parser.parse_args()
    aml_logging.setup(args.logtype)
    aml_logging.debug('This is a debug message')
    aml_logging.info('This is an info message')
    aml_logging.warning('This is a warning message')
    aml_logging.error('This is an error message')
    aml_logging.critical('This is a critical error message')


if __name__ == '__main__':
    main()