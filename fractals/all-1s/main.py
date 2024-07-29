'''

an 8x8 board with all off except an odd number of tiles, 
the challenge is to design rules so that all cells turn on eventually 
if the starting condition has an even number of switched on then it will not converge

'''


import matplotlib.pyplot as plt 
import numpy as np

def init_board(initial_conditions):
    board = np.zeros((8,8))
    for xy in initial_conditions:
        board[xy[0]][xy[1]] = 1

    return board

def apply_rules(board):

    return board

def check_convergence(board):
    return (board==np.ones((8,8)))

def main():

    # select the starting tiles to be 1, all other tiles are zero
    initial_conditions = [(1,1)]

    board = init_board(initial_conditions)
    
    converged = 0
    iteration = 0
    max_iteration = 100

    while (not converged) and (iteration != max_iteration):
        board = apply_rules(board)
        plt.imshow(board)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        converged = check_convergence(board)
        iteration = iteration+1
        stop


    print("Congerged!")





if __name__ == '__main__':
    main()
