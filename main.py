import sys
import pygame
import threading
import numpy as np

from driver import NeuralNetwork
from gui import *

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 860, 660
CELLS = 28
GRID_SIZE = 560
CELL_SIZE = GRID_SIZE // CELLS
WHITE = (255, 255, 255)
BLACK = (28, 28, 28)
THEME_COLOUR = (38, 58, 76)

# Initialise Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Neural Network Visualiser")

# Initialise the drawing grid and other variables
drawing_grid = np.zeros((CELLS, CELLS), dtype=int)
clock = pygame.time.Clock()
previous_pos = None
prev_value = 0
title = True
controls = False
selected_epochs = False
is_drawing = False
is_alive = False

nn = NeuralNetwork()
ui_manager = UIManager([percentage_handler:= PercentageHandler(),
                       epoch_slider:= Slider((25, 580), 20, 1),
                       wait_time_slider:= Slider((25, 620), 1000, 0, "ms"),
                       accuracy:= TextBox("UNTRAINED", (385, 620), colour="red"),
                       epoch:= TextBox("EPOCH 0/1", (385, 580), target_word="EPOCH")])


def train_neural_network(epochs: int) -> None:
    """
    Train the neural network for the specified number of epochs.
    """
    nn.train(epochs)
    drawing_grid.fill(0)


def draw_smooth_line(start_pos: tuple, end_pos: tuple, colour: int, training: bool, brush_size=10) -> None:
    """
    Draw a smooth line on the grid.

    Args:
        start_pos (tuple): The starting position of the line.
        end_pos (tuple): The ending position of the line.
        colour (int): The color value (0 or 1) of the line.
        training (bool): True if the network is training, False otherwise.
        brush_size (int, optional): The size of the brush. Defaults to 10.
    """
    if training:
        return

    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    distance = max(abs(dx), abs(dy))
    if distance == 0:
        distance = 1

    for i in range(distance + 1):
        x = start_pos[0] + int(i * dx / distance)
        y = start_pos[1] + int(i * dy / distance)

        for j in range(-brush_size, brush_size + 1):
            for k in range(-brush_size, brush_size + 1):
                row, col = (x + j) // CELL_SIZE, (y + k) // CELL_SIZE
                if 0 <= row < CELLS and 0 <= col < CELLS:
                    drawing_grid[col][row] = colour


def draw_grid(screen) -> None:
    """
    Draw the grid and the content of the drawing grid on the screen.
    """
    for row in range(CELLS):
        for col in range(CELLS):
            greyscale = drawing_grid[col][row] * 255
            if greyscale <= 28:
                pygame.draw.rect(
                    screen, BLACK, (row * CELL_SIZE, col * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            else:
                pygame.draw.rect(screen, (greyscale, greyscale, greyscale),
                                 (row * CELL_SIZE, col * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for x in range(0, GRID_SIZE + 1, CELL_SIZE):
        pygame.draw.line(screen, WHITE, (x, 0), (x, GRID_SIZE))
    for y in range(0, GRID_SIZE + 1, CELL_SIZE):
        pygame.draw.line(screen, WHITE, (0, y), (GRID_SIZE, y))


def title_screen(screen) -> bool:
    """
    Display the title screen and wait for the user to press enter.

    Returns:
        bool: True if the user presses enter, False otherwise.
    """
    boxes = [
        TextBox("Neural Network Visualiser", (100, SCREEN_HEIGHT //
                2 - 200), font_size=60, colour=THEME_COLOUR),
        TextBox("Trained on the MNIST database",
                (150, SCREEN_HEIGHT // 2 - 100), font_size=45),
        TextBox("By vSparkyy", (350, SCREEN_HEIGHT // 2),
                colour=THEME_COLOUR),
        TextBox("Assets by Jarmishan", (335, SCREEN_HEIGHT // 2 + 25),
                colour=THEME_COLOUR, font_size=20),
        TextBox("[PRESS ENTER AT ANY TIME FOR CONTROLS...]",
                (85, SCREEN_HEIGHT // 2 + 200), font_size=40, target_word="ENTER")
    ]
    start_time = pygame.time.get_ticks()
    title_duration = 4000
    exit_screen = False

    while pygame.time.get_ticks() - start_time < title_duration and not exit_screen:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    control_screen(screen)
                    exit_screen = True

        for text_box in boxes:
            text_box.update(screen)

        pygame.display.flip()

    return False


def control_screen(screen) -> bool:
    """
    Display the controls screen and wait for the user to press enter.

    Returns:
        bool: True if the user presses enter, False otherwise.
    """
    text_boxes = [
        TextBox("R - Reset grid", (300, SCREEN_HEIGHT // 2 - 275),
                font_size=40, target_word="R"),
        TextBox("Space - Begin training", (250, SCREEN_HEIGHT // 2 - 175),
                font_size=40, target_word="Space"),
        TextBox("C - Unlock epoch slider (to train again)", (100, SCREEN_HEIGHT // 2 - 75),
                font_size=40, target_word="C"),
        TextBox("G - Generate random noise", (200, SCREEN_HEIGHT // 2 + 15),
                font_size=40, target_word="G"),
        TextBox("Left/Right Click - Draw/Erase", (180, SCREEN_HEIGHT // 2 + 115),
                font_size=40, target_word="Left/Right"),
        TextBox("Press enter again to exit this screen...", (185, SCREEN_HEIGHT // 2 + 215),
                target_word="enter")
    ]
    exit_screen = False

    while not exit_screen:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    exit_screen = True

        screen.fill(BLACK)
        for text_box in text_boxes:
            text_box.update(screen)
        pygame.display.flip()

    return False


while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                drawing_grid.fill(0)

            if event.key == pygame.K_SPACE and not selected_epochs:
                selected_epochs = True
                training_thread = threading.Thread(
                    target=train_neural_network, args=[int(epoch_slider.current_value)])
                training_thread.start()

            if event.key == pygame.K_c and selected_epochs:
                prev_value = int(epoch_slider.current_value) + prev_value
                selected_epochs = False
                epoch_slider.lock("off")

            if event.key == pygame.K_RETURN:
                controls = True

            if event.key == pygame.K_g and selected_epochs and not is_alive:
                drawing_grid = np.random.uniform(0, 1, (28, 28)) ** 2

            if event.key == pygame.K_k:
                print(f"previous value: {prev_value}\nepoch slider current value: {int(epoch_slider.current_value)}\ncurrent epoch neural network: {nn.current_epoch}\n")


        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 or event.button == 3:
                brush_colour = int(not event.button == 3)
                is_drawing = True
                previous_pos = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 or event.button == 3:
                is_drawing = False
                previous_pos = None

        if is_drawing and selected_epochs:
            current_pos = pygame.mouse.get_pos()
            if previous_pos:
                draw_smooth_line(previous_pos, current_pos,
                                 brush_colour, training_thread.is_alive())
            previous_pos = current_pos

    if selected_epochs:
        epoch_slider.lock("on")
        is_alive = training_thread.is_alive()
        if nn.img is not None and is_alive:
            drawing_grid = np.copy(nn.img.reshape(28, 28))
            percentage_handler.percentage_data(
                nn.get_percentages(nn.layers[-1].output))
            pygame.time.wait(int(wait_time_slider.current_value))
            accuracy.colour = (0, 163, 73)
            accuracy.text = f"{'TRAINING...' if not nn.accuracy else nn.accuracy + '% ACCURACY'}"

        if not is_alive:
            accuracy.text = f"{nn.accuracy}% ACCURACY"
            percentage_handler.percentage_data(nn.get_percentages(nn.image))
            nn.test(drawing_grid.flatten().reshape(-1, 1))

    epoch.text = f"EPOCH {nn.current_epoch}/{int(epoch_slider.current_value) + prev_value}"

    screen.fill(BLACK)
    if title:
        title = title_screen(screen)
    if controls:
        controls = control_screen(screen)
    draw_grid(screen)
    ui_manager.draw_elements(screen)

    pygame.display.flip()
    clock.tick(60)
