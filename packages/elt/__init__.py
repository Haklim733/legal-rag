""" inserting logging for package """
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
