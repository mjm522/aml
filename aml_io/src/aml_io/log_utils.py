import logging


class aml_logging(object):

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
