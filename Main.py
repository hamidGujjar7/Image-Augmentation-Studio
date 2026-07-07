"""
main.py
-----------------------------------------------------------------------------
Entry point for the Image Augmentation Studio.

Run with:
    python main.py
-----------------------------------------------------------------------------
"""

from Src.gui_app import ImageAugmentationApp


def main():
    app = ImageAugmentationApp()
    app.mainloop()


if __name__ == "__main__":
    main()