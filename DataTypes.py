# -*- coding: utf-8 -*-

from enum import Enum


class SquareType(Enum):
    WHITE = 0
    BLACK = 1
    EMPTY = 2
    ERROR = 3


class GameStatus(Enum):
    WHITE_WINS = 0
    BLACK_WINS = 1
    DRAW = 2
    PLAYING = 3
    ERROR = 4
