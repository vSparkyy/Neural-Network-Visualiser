import os

import pygame
import numpy as np

# Constants
COLOURS = {
    'BAR': (38, 58, 76),
    'WHITE': (200, 200, 200),
    'GREEN': (0, 163, 73)
}
BAR_WIDTH = 250
BAR_HEIGHT = 24
PIXEL_FONT_SIZE = 30

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ASSETS = {
    'BAR_FRAME': os.path.join(BASE_PATH, "assets/bar_frame.png"),
    'SLIDER': os.path.join(BASE_PATH, "assets/slider.png"),
    'PIXEL_FONT': os.path.join(BASE_PATH, "assets/pixel_font.ttf"),
}

# Fonts and Images
pygame.init()
FONT = pygame.font.Font(ASSETS['PIXEL_FONT'], PIXEL_FONT_SIZE)
BAR_FRAME_IMAGE = pygame.image.load(ASSETS['BAR_FRAME'])
SLIDER_IMAGE = pygame.image.load(ASSETS['SLIDER'])


class UIManager:
    """Class to manage UI elements."""

    def __init__(self, elements):
        self.elements = elements

    def draw_elements(self, surface):
        """Draw all elements on the given surface."""
        for element in self.elements:
            element.update(surface)


class Slider:
    """Class representing a slider"""

    def __init__(self, position: tuple, max_value: int, min_value=0, extra_text=""):
        """
        Initialise slider on surface

        Args:
            position (tuple): The (x, y) position of the slider.
            max_value (int): Maximum value that the slider can reach.
            min_value (int, optional): Minimum value that the slider can reach.
            extra_text (str, optional): Any extra text to be displayed next to the slider's value.
        """
        self.max_value = max_value
        self.min_value = min_value
        self.pressed = False
        self.extra_text = extra_text
        self.current_value = min_value
        self.slider_width = 24
        self.x = position[0]
        self.y = position[1]
        self.slider_rect = pygame.Rect(self.x, self.y, BAR_WIDTH, BAR_HEIGHT)
        self.locked = False

    def lock(self, switch):
        """Lock or unlock the slider."""
        self.locked = switch == "on"

    def update(self, surface):
        """Update the slider's position and display."""
        if not self.locked:
            mouse_pos = pygame.mouse.get_pos()
            buttons = pygame.mouse.get_pressed()
            if buttons[0]:
                if self.slider_rect.collidepoint(mouse_pos):
                    self.pressed = True
                    new_value = (mouse_pos[0] - self.x) * (self.max_value -
                                                           self.min_value) / (BAR_WIDTH - self.slider_width)
                    self.current_value = max(
                        self.min_value, min(new_value, self.max_value))
            else:
                self.pressed = False

        surface.blit(BAR_FRAME_IMAGE, (self.x, self.y))
        value_text = f"{str(int(self.current_value)) + self.extra_text}" if self.extra_text else str(
            int(self.current_value))
        value_font = FONT.render(value_text, False, COLOURS['WHITE'])
        slider_pos = self.x + ((self.current_value - self.min_value) / (
            self.max_value - self.min_value)) * (BAR_WIDTH - self.slider_width)
        slider_pos = max(self.x, min(slider_pos, self.x +
                         BAR_WIDTH - self.slider_width))
        surface.blit(SLIDER_IMAGE, (slider_pos, self.y))
        surface.blit(value_font, (self.x + 110, self.y - 1))


class TextBox:
    """Class representing a text box."""

    def __init__(self, text: str, position: tuple, font_size=30, colour=COLOURS['WHITE'], target_word="", target_colour=COLOURS['BAR']):
        """
        Initialise text box on surface

        Args:
            text (str): Text to be displayed on text box.
            position (tuple): The (x, y) position of the text box.
            font_size (int): Font size of text.
            colour (tuple, optional): (R, G, B) colour for text.
            target_word (str, optional): A word within the text defined to highlight.
            target_colour (tuple, optional): Target colour for word to highlight.
        """
        self.font = pygame.font.Font(ASSETS['PIXEL_FONT'], font_size)
        self.text = text
        self.visible = True
        self.colour = colour
        self.x = position[0]
        self.y = position[1]
        self.target_word = target_word
        self.target_colour = target_colour
        self.cached_surfaces = {}

    def update(self, surface):
        """Update the text box and display."""
        if not self.visible:
            return

        words = self.text.split(" ")
        space_width = self.font.size(" ")[0]

        x_offset = self.x
        y_offset = self.y

        for word in words:
            if word in self.cached_surfaces:
                text_surface = self.cached_surfaces[word]
            else:
                text_surface = self.font.render(word, False, self.colour)
                self.cached_surfaces[word] = text_surface

            word_width = text_surface.get_width()

            if word == self.target_word and self.target_colour is not None:
                word_surface = self.font.render(
                    word, False, self.target_colour)
                surface.blit(word_surface, (x_offset, y_offset))
            else:
                surface.blit(text_surface, (x_offset, y_offset))

            x_offset += word_width + space_width
