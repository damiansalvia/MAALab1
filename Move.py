# -*- coding: utf-8 -*-


class Move(object):

    def __init__(self, row, col):
        self._row = row
        self._col = col

    def get_row(self):
        return self._row

    def get_col(self):
        return self._col
