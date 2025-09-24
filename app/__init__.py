"""
DSP Final Project - Audio Algorithm Comparison App

A GUI application for comparing audio algorithms that modify tempo.
"""

from tkinter import Tk
from .menu import MainMenu
import logging

def main():
    """
    Launch the main GUI application.
    
    This function creates the main Tkinter window and starts the application.
    """
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Started')
    
    root = Tk()
    app = MainMenu(root)
    root.mainloop()
    
    logger.info('Finished')

__version__ = "0.1.0"
__author__ = "DSP Final Project Team"
