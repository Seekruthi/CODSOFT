import tkinter as tk

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("TicTacToe")

        self.root.minsize(400, 400)
        self.root.maxsize(800, 800)

        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.Player = 'O'
        self.Computer = 'X'
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_board()

        self.message_label = tk.Label(self.root, text="", font=("comicsans", 24))
        self.message_label.grid(row=3, column=0, columnspan=3)

        self.create_control_buttons()
        self.ai_firstGuess()

        for i in range(3):
            self.root.grid_rowconfigure(i, weight=1)  
            self.root.grid_columnconfigure(i, weight=1)

    def create_board(self):
        for row in range(3):
            for col in range(3):
                button = tk.Button(self.root, text=" ", font=("comicsans", 24), height=3, width=6,
                                   command=lambda row=row, col=col: self.button_onClick(row, col),
                                   bg="#f0f0f0", fg="black")  
                button.grid(row=row, column=col, sticky="nsew")  
                self.buttons[row][col] = button

    def button_onClick(self, row, col):
        if self.buttons[row][col]["text"] == " ":
            self.updateBoard(row, col, self.Player)

            if not self.game_over():  
                ai_guess = self.aiGuess(self.Computer, self.Player)
                if ai_guess:
                    self.updateBoard(ai_guess[0], ai_guess[1], self.Computer)

            self.game_over()

    def disable_buttons(self):
        for row in range(3):
            for col in range(3):
                self.buttons[row][col]["state"] = tk.DISABLED

    def reset(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        for row in range(3):
            for col in range(3):
                self.buttons[row][col]["text"] = " "
                self.buttons[row][col]["state"] = tk.NORMAL  
                self.buttons[row][col]["bg"] = "#f0f0f0" 
        self.message_label.config(text="")  

    def create_control_buttons(self):
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=4, column=0, columnspan=3, sticky="ew")

        reset_button = tk.Button(control_frame, text="Reset", font=("comicsans", 16), command=self.reset)
        reset_button.pack(side=tk.LEFT, padx=5, expand=True)  

        exit_button = tk.Button(control_frame, text="Exit", font=("comicsans", 16), command=self.root.destroy)
        exit_button.pack(side=tk.LEFT, padx=5, expand=True)  

    def updateBoard(self, row, col, symbol):
        self.board[row][col] = symbol
        self.buttons[row][col]["text"] = symbol
        if symbol == self.Player:
            self.buttons[row][col]["bg"] = "lightblue"  
        else:
            self.buttons[row][col]["bg"] = "salmon" 

    def ai_firstGuess(self):
        ai_guess = self.aiGuess(self.Computer, self.Player)
        if ai_guess:
            self.updateBoard(ai_guess[0], ai_guess[1], self.Computer)

    def aiGuess(self, Computer, Player):
        best_move = None
        max_score = -float('inf')
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == ' ':
                    self.board[row][col] = Computer
                    score = self.minimaxAlgo(self.board, 0, False, -float('inf'), float('inf'), Computer, Player)
                    self.board[row][col] = ' '
                    if score > max_score:
                        max_score = score
                        best_move = (row, col)
        return best_move

    def winner(self, symbol):
        for row in range(3):
            if all(self.board[row][col] == symbol for col in range(3)):
                return True
        for col in range(3):
            if all(self.board[row][col] == symbol for row in range(3)):
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == symbol or \
           self.board[0][2] == self.board[1][1] == self.board[2][0] == symbol:
            return True
        return False

    def draw(self):
        return all(self.board[row][col] != ' ' for row in range(3) for col in range(3))

    def game_over(self):
        if self.winner(self.Player):
            self.message_label.config(text="You Won!")
            self.disable_buttons()
            return True
        elif self.winner(self.Computer):
            self.message_label.config(text="You Lost!")
            self.disable_buttons()
            return True
        elif self.draw():
            self.message_label.config(text="Draw!")
            self.disable_buttons()
            return True
        return False

    def minimaxAlgo(self, board, depth, is_maximizing, alpha, beta, Computer, Player):
        if self.winner(Computer):
            return 1
        if self.winner(Player):
            return -1
        if self.draw():
            return 0
        if is_maximizing:
            max_score = -float('inf')
            for row in range(3):
                for col in range(3):
                    if board[row][col] == ' ':
                        board[row][col] = Computer
                        score = self.minimaxAlgo(board, depth + 1, False, alpha, beta, Computer, Player)
                        board[row][col] = ' '
                        max_score = max(score, max_score)
                        alpha = max(alpha, score)
                        if alpha >= beta:
                            break
            return max_score
        else:
            min_score = float('inf')
            for row in range(3):
                for col in range(3):
                    if board[row][col] == ' ':
                        board[row][col] = Player
                        score = self.minimaxAlgo(board, depth + 1, True, alpha, beta, Computer, Player)
                        board[row][col] = ' '
                        min_score = min(score, min_score)
                        beta = min(beta, score)
                        if alpha >= beta:
                            break
            return min_score

root = tk.Tk()
game = TicTacToe(root)
root.mainloop()
