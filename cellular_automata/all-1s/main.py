'''

an 8x8 board with all off except an odd number of tiles, 
the challenge is to design rules so that all cells turn on eventually 
if the starting condition has an even number of switched on then it will not converge

'''


import matplotlib.pyplot as plt 
import numpy as np

def init_board(initial_conditions,board_size):
    board = np.zeros((board_size,board_size))
    for xy in initial_conditions:
        board[xy[0]][xy[1]] = 1

    return board

def apply_rules(board):

    for y in range(len(board[0])):
        for x in range(len(board)):

            if board[x][y]==1:
                if x>0 and y>0 and x<len(board)-1 and y<len(board[0])-1:
                    print(f"central case: ({x},{y})")

                    board[x-1][y+1] = 1
                    board[x+1][y+1] = 1

                    board[x-1][y-1] = 1
                    board[x+1][y-1] = 1
                elif y==0 and x>0 and x<len(board)-1:
                    print(f"left wall, not including corners: ({x},{y})")

                    board[x-1][y+1] = 1
                    board[x+1][y+1] = 1
                elif y==0 and x==0:
                    print(f"bottom left corner")
                    board[x+1][y+1] = 1
                elif y==len(board[0]) and x>=0 and x<len(board)-1:
                    print(f"right wall not including corners: ({x},{y})")


                    board[x-1][y-1] = 1
                    board[x+1][y-1] = 1
                elif y==len(board[0]) and x==len(board):
                    print(f"top right corner: ({x},{y})")
                    board[x-1][y-1] = 1
                elif y==0 and x==len(board):
                    print(f"top left corner: ({x},{y})")
                    board[x-1][y+1] = 1
                elif y==len(board[0]) and x==0:
                    print(f"bottom left corner: ({x},{y})")
                    board[x+1][y-1] = 1

            if board[x][y]==0:
                neighbors = 0
                if x>0 and y>0 and x<len(board)-1 and y<len(board[0])-1:
                    print(f"central case: ({x},{y})")
                    if board[x+1][y] == 1:
                        neighbors = neighbors + 1
                    if board[x-1][y] == 1: 
                        neighbors = neighbors + 1
                    if board[x][y-1] == 1:
                        neighbors = neighbors + 1
                    if board[x][y+1] == 1:
                        neighbors = neighbors + 1

                elif y==0 and x>0 and x<len(board)-1:
                    print(f"left wall, not including corners: ({x},{y})")
                    if board[x-1][y] == 1:
                        neighbors = neighbors + 1
                    if board[x+1][y] == 1:
                        neighbors = neighbors + 1
                    if board[x][y+1] == 1:
                        neighbors = neighbors + 1

                elif y==0 and x==0:
                    print(f"bottom left corner")
                    if board[x+1][y] == 1:
                        neighbors = neighbors + 1
                    if board[x][y+1] == 1:
                        neighbors = neighbors + 1

                elif y==len(board[0]) and x>=0 and x<len(board)-1:
                    print(f"right wall not including corners: ({x},{y})")
                    if board[x-1][y] == 1:
                        neighbors = neighbors + 1
                    if board[x+1][y] == 1:
                        neighbors = neighbors + 1
                    if board[x][y-1] == 1:
                        neighbors = neighbors + 1

                elif y==len(board[0]) and x==len(board):
                    print(f"top right corner: ({x},{y})")
                    if board[x-1][y] == 1:
                        neighbors = neighbors + 1
                    if board[x][y-1] == 1:
                        neighbors = neighbors + 1

                elif y==0 and x==len(board):
                    print(f"top left corner: ({x},{y})")
                    if board[x-1][y] == 1:
                        neighbors = neighbors + 1
                    if board[x][y+1] == 1:
                        neighbors = neighbors + 1

                elif y==len(board[0]) and x==0:
                    print(f"bottom left corner: ({x},{y})")
                    if board[x+1][y] == 1:
                        neighbors = neighbors + 1
                    if board[x][y-1] == 1:
                        neighbors = neighbors + 1
                
                if neighbors>1:
                    board[x][y] = 1


    print(board)
    return board

def check_convergence(board,board_size):
    test = 1

    for y in range(len(board[0])):
        for x in range(len(board)):
            test = (board[x][y]==1)
            if test == 0:
                return test

    return test

def main():

    board_size = 10

    # select the starting tiles to be 1, all other tiles are zero
    initial_conditions = [(2,1)]

    board = init_board(initial_conditions,board_size)
    
    converged = 0
    iteration = 0
    max_iteration = 10

    while (not converged) and (iteration != max_iteration):
        board = apply_rules(board)
        plt.imshow(board)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        converged = check_convergence(board,board_size)
        iteration = iteration+1



    print(f"Congerged! Iteration {iteration}")





if __name__ == '__main__':
    main()
