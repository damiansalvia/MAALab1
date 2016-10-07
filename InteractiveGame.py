# -*- coding: UTF-8 -*-

from Tkinter import *
from Board import Board
from DataTypes import SquareType, GameStatus
from Move import Move
from Game import Game
from copy import deepcopy


DEFAULT_FONT = ('Helvetica', 13)
GridSize = 60  # size in pixels of each square on playing board
PieceSize = GridSize - 8  # size in pixels of each playing piece
BoardColor = '#008000'  # color of board - medium green
HighlightColor = '#00a000'
Offset = 2  # offset in pixels of board from edge of canvas
PlayerNames = {SquareType.EMPTY: '',
               SquareType.BLACK: 'Negras',
               SquareType.WHITE: 'Blancas'}  # Names of players as displayed to the user
PlayerColors = {SquareType.EMPTY: '',
                SquareType.BLACK: '#000000',
                SquareType.WHITE: '#ffffff'}  # rgb values for black, white
MoveDelay = 500


class InteractiveGame(Game):

    class Square:
        """Holds data related to a square of the board"""
        def __init__(self, x, y):
            self.x, self.y = x, y  # location of square (in range 0-7)
            self.player = SquareType.EMPTY  # number of player occupying square
            self.squareId = 0  # canvas id of rectangle
            self.pieceId = 0  # canvas id of circle

    def __init__(self, players=()):
        """Initialize the interactive game board.  An optional list of
           computer opponent strategies can be supplied which will be
           displayed in a menu to the user.
        """
        # create a Tk frame to hold the gui
        self._frame = Frame()
        # set the window title
        self._frame.master.wm_title('Othello')
        # build the board on a Tk drawing canvas
        size = 8 * GridSize  # make room for 8x8 squares
        self._canvas = Canvas(self._frame, width=size, height=size)
        self._canvas.pack()
        # add button for starting game
        self._menuFrame = Frame(self._frame)
        self._menuFrame.pack(expand=Y, fill=X)
        self._newGameButton = Button(self._menuFrame, text='Nuevo Juego', command=self._new_game)
        self._newGameButton.pack(side=LEFT, padx=5)
        Label(self._menuFrame).pack(side=LEFT, expand=Y, fill=X)
        # add menus for choosing player strategies
        self._players = {}  # strategies, indexed by name
        option_menu_args = [self._menuFrame, 0, 'Humano']
        for player in players:
            name = player.name
            option_menu_args.append(name)
            self._players[name] = player
        self._strategyVars = {SquareType.EMPTY: ''}  # dummy entry so strategy indexes match player numbers
        # make an menu for each player
        for n in (SquareType.BLACK, SquareType.WHITE):
            label = Label(self._menuFrame, anchor=E, text='%s:' % PlayerNames[n])
            label.pack(side=LEFT, padx=10)
            var = StringVar()
            var.set('Humano')
            # var.trace('w', self._player_menu_callback)
            self._strategyVars[n] = var
            option_menu_args[1] = var
            menu = apply(OptionMenu, option_menu_args)
            menu.pack(side=LEFT)
        # add a label for showing the status
        self._status = Label(self._frame, relief=SUNKEN, anchor=W)
        self._status.pack(expand=Y, fill=X)
        # map the frame in the main Tk window
        self._frame.pack()
        # track the board state
        self._squares = {}  # Squares indexed by (x,y)
        self._enabledSpaces = ()  # list of valid moves as returned by Board.get_possible_moves()
        for x in xrange(8):
            for y in xrange(8):
                square = self._squares[x, y] = InteractiveGame.Square(x, y)
                x0 = x * GridSize + Offset
                y0 = y * GridSize + Offset
                square.squareId = self._canvas.create_rectangle(x0, y0,
                                                                x0 + GridSize, y0 + GridSize,
                                                                fill=BoardColor)

        # _afterId tracks the current 'after' proc so it can be cancelled if needed
        self._afterId = 0

        # ready to go - start a new game!
        super(InteractiveGame, self).__init__()

    def _new_game(self):
        self._game_status = GameStatus.PLAYING
        self._move_list = []
        self._active_players = {}
        for color in (SquareType.BLACK, SquareType.WHITE):
            # noinspection PyUnresolvedReferences
            self._active_players[color] = self._players.get(self._strategyVars[color].get())
            if self._active_players[color]:
                self._active_players[color] = self._active_players[color](color)
        self._last_move = None
        # delete existing pieces
        for s in self._squares.values():
            if s.pieceId:
                self._canvas.delete(s.pieceId)
                s.pieceId = 0
        # create a new board state and display it
        self._state = Board(8, 8)
        self._turn = SquareType.BLACK
        self._update_board()

    def play(self):
        """Play the game! (this is the only public method)"""
        self._frame.mainloop()

    def _update_board(self):
        # reset any enabled spaces
        self._disable_spaces()
        # cancel 'after' proc, if any
        if self._afterId:
            self._frame.after_cancel(self._afterId)
            self._afterId = 0
        # update canvas display to match current state
        for x in xrange(8):
            for y in xrange(8):
                square = self._squares[(x, y)]
                if square.pieceId:
                    if square.player != self._state.get_position(x, y):
                        self._canvas.itemconfigure(square.pieceId, fill=PlayerColors[self._state.get_position(x, y)])
                else:
                    if self._state.get_position(x, y) != SquareType.EMPTY:
                        x0 = x * GridSize + Offset + 4
                        y0 = y * GridSize + Offset + 4
                        square.pieceId = self._canvas.create_oval(x0, y0,
                                                                  x0 + PieceSize, y0 + PieceSize,
                                                                  fill=PlayerColors[self._state.get_position(x, y)])

        ai = self._active_players[self._turn]
        if self._game_status == GameStatus.PLAYING:
            if ai:
                self._process_ai(ai)
            else:
                self._process_human()
        else:
            self._game_over()

    def _process_ai(self, ai):
        if self._state.get_possible_moves(self._turn):
            self._last_move = ai.move(deepcopy(self._state), self._last_move)
            self._do_move(self._last_move, self._turn)
        else:
            self._last_move = None
        self._pass_turn()
        self._afterId = self._frame.after(MoveDelay, self._update_board)

    def _process_human(self):
        if self._state.get_possible_moves(self._turn):
            self._enabledSpaces = self._state.get_possible_moves(self._turn)
            self._enable_spaces(self._turn)
        else:
            self._last_move = None
            self._pass_turn()
            self._update_board()

    def _select_space(self, x, y, color):
        self._last_move = Move(x, y)
        self._do_move(self._last_move, color)
        self._pass_turn()
        self._update_board()

    def _enable_spaces(self, color):
        # make spaces active where a legal move is possible (only used for human players)
        for move in self._enabledSpaces:
            row = move.get_row()
            col = move.get_col()
            square_id = self._squares[row, col].squareId
            self._canvas.tag_bind(square_id, '<ButtonPress>',
                                  lambda e, x=row, y=col, _color=color: self._select_space(x, y, _color))
            self._canvas.tag_bind(square_id, '<Enter>',
                                  lambda e, c=self._canvas, _id=square_id: c.itemconfigure(_id, fill=HighlightColor))
            self._canvas.tag_bind(square_id, '<Leave>',
                                  lambda e, c=self._canvas, _id=square_id: c.itemconfigure(_id, fill=BoardColor))

    def _disable_spaces(self):
        # remove event handlers for all enabled spaces
        for move in self._enabledSpaces:
            x = move.get_row()
            y = move.get_col()
            if x == -1:
                break
            square_id = self._squares[x, y].squareId
            self._canvas.tag_unbind(square_id, '<ButtonPress>')
            self._canvas.tag_unbind(square_id, '<Enter>')
            self._canvas.tag_unbind(square_id, '<Leave>')
            self._canvas.itemconfigure(square_id, fill=BoardColor)
        self._enabledSpaces = ()

    def _game_over(self):
        self._log_to_file()
        top = Toplevel()
        top.title('Fin del juego')
        msg_text = 'El juego finalizó por un error.'
        if self._game_status == GameStatus.WHITE_WINS:
            msg_text = 'Gana el jugador con las piezas blancas.'
        elif self._game_status == GameStatus.BLACK_WINS:
            msg_text = 'Gana el jugador con las piezas negras.'
        elif self._game_status == GameStatus.DRAW:
            msg_text = 'El juego terminó en empate.'
        pop_up_msg = Message(top, text=msg_text)
        pop_up_msg.pack()

        button = Button(top, text='Ok', command=top.destroy)
        button.pack()

if __name__ == '__main__':
    from Players.RandomPlayer import RandomPlayer
    from Players.GreedyPlayer import GreedyPlayer
    t = InteractiveGame([RandomPlayer, GreedyPlayer])
    t.play()
