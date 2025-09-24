from menu import MainMenu
from tkinter import Tk

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logger.info('Started')
    root = Tk()
    app = MainMenu(root)
    root.mainloop()
    logger.info('Finished')