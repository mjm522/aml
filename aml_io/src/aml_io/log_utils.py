import logging

LEVELS = {'debug':    logging.DEBUG,
          'info':     logging.INFO,
          'warning':  logging.WARNING,
          'error':    logging.ERROR,
          'critical': logging.CRITICAL}

class aml_logging(object):

	# @classmethod
	# def setup(cls, level_name):
	# 	if level_name is None:
	# 		level_name = 'info'
	# 	level = LEVELS.get(level_name, logging.NOTSET)
 #    	logging.basicConfig(level=level)

	@classmethod
	def debug(cls, msg):
		logging.debug(msg);

	@classmethod
	def info(cls, msg):
		logging.info(msg);

	@classmethod
	def warning(cls, msg):
		logging.warning(msg);

	@classmethod
	def error(cls, msg):
		logging.info(msg);

	@classmethod
	def critical(cls, msg):
		logging.info(msg);
