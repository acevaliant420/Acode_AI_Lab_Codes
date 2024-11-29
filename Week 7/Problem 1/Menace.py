from collections import Counter
import random
import json
import os

class GameBoard:
    def __init__(self):
        self.grid = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

    def __str__(self):
        return ("\n 0 || 1 || 2 \t %s || %s || %s \n 3 || 4 || 5\t %s || %s || %s \n 6 || 7 || 8\t %s || %s || %s " %
                (self.grid[0], self.grid[1], self.grid[2], self.grid[3], self.grid[4], self.grid[5], self.grid[6], self.grid[7], self.grid[8]))

    def isValidMove(self, position):
        try:
            position = int(position)
        except ValueError:
            return False
        return 0 <= position <= 8 and self.grid[position] == " "

    def checkWin(self):
        return ((self.grid[0] != ' ' and
                 ((self.grid[0] == self.grid[1] == self.grid[2]) or
                  (self.grid[0] == self.grid[3] == self.grid[6]) or
                  (self.grid[0] == self.grid[4] == self.grid[8])) )
                or (self.grid[4] != ' ' and
                    ((self.grid[1] == self.grid[4] == self.grid[7]) or
                    (self.grid[3] == self.grid[4] == self.grid[5]) or
                    (self.grid[2] == self.grid[4] == self.grid[6])) )
                or (self.grid[8] != ' ' and
                    ((self.grid[2] == self.grid[5] == self.grid[8]) or
                    (self.grid[6] == self.grid[7] == self.grid[8])) ))

    def checkDraw(self):
        return all((x != " " for x in self.grid))

    def makeMove(self, spot, player):
        self.grid[spot] = player

    def getGridState(self):
        return ''.join(self.grid)

class LearningBot:
    def __init__(self):
        self.boxes = self.load_states()
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.movesHistory = []

    def load_states(self):
        if os.path.exists('states.json'):
            with open('states.json', 'r') as f:
                return json.load(f)
        else:
            # 10 predefined initial states
            return {
                'XOXOX    ': [5, 6, 7, 8],
                'XO XO X  ': [3, 7, 8],
                'XXO OX   ': [5, 6, 7],
                ' OXOXOX  ': [0, 7, 8],
                'OXOXOX   ': [6, 7, 8],
                'XOXOXOXO ': [8],
                'X X O O X': [1, 7, 8],
                'XX O OOX ': [3, 7],
                'O OX XOX ': [1, 3, 7],
                'XXOX O O ': [6, 8]
            }

    def save_states(self):
        with open('states.json', 'w') as f:
            json.dump(self.boxes, f)

    def newGame(self):
        self.movesHistory = []

    def selectMove(self, game_board):
        board_state = game_board.getGridState()

        if board_state not in self.boxes:
            available_moves = [i for i, mark in enumerate(board_state) if mark == " "]
            self.boxes[board_state] = available_moves * ((len(available_moves) + 2) // 2)

        possible_moves = self.boxes[board_state]

        if len(possible_moves):
            chosen_move = random.choice(possible_moves)
            self.movesHistory.append((board_state, chosen_move))
        else:
            chosen_move = -1
        return chosen_move

    def recordWin(self):
        for (board_state, move) in self.movesHistory:
            self.boxes[board_state].extend([move, move, move])
        self.wins += 1

    def recordDraw(self):
        for (board_state, move) in self.movesHistory:
            self.boxes[board_state].append(move)
        self.draws += 1

    def recordLoss(self):
        for (board_state, move) in self.movesHistory:
            matchbox = self.boxes[board_state]
            del matchbox[matchbox.index(move)]
        self.losses += 1

    def stateCount(self):
        return len(self.boxes)

class Player:
    def __init__(self):
        pass

    def newGame(self):
        print("Ready to play!")

    def selectMove(self, game_board):
        while True:
            move = input("Enter your move: ")
            if game_board.isValidMove(move):
                break
            print("Invalid move.")
        return int(move)

    def recordWin(self):
        print("You won!")

    def recordDraw(self):
        print("It's a draw.")

    def recordLoss(self):
        print("You lost.")

def startGame(first_player, second_player, silent=False):
    first_player.newGame()
    second_player.newGame()
    game_board = GameBoard()

    if not silent:
        print("\nNew game started!")
        print(game_board)

    while True:
        move = first_player.selectMove(game_board)
        if move == -1:
            if not silent:
                print("Player resigned.")
            first_player.recordLoss()
            second_player.recordWin()
            break

        game_board.makeMove(move, 'X')

        if not silent:
            print(game_board)
        if game_board.checkWin():
            first_player.recordWin()
            second_player.recordLoss()
            break
        if game_board.checkDraw():
            first_player.recordDraw()
            second_player.recordDraw()
            break

        move = second_player.selectMove(game_board)

        if move == -1:
            if not silent:
                print("Player resigned.")
            second_player.recordLoss()
            first_player.recordWin()
            break

        game_board.makeMove(move, 'O')

        if not silent:
            print(game_board)
        if game_board.checkWin():
            second_player.recordWin()
            first_player.recordLoss()
            break

    # Save the bot's learned states at the end of the game
    first_player.save_states()

if __name__ == '__main__':
    bot1 = LearningBot()
    human_player = Player()

    print("Enter 1 to use a trained model or 0 for a new game.")
    option = int(input())

    if option == 0:
        startGame(bot1, human_player)
    
    elif option == 1:
        print("Total states in memory:", bot1.stateCount())
        startGame(bot1, human_player)

