import pygame

def init():
    pygame.init()
    win = pygame.display.set_mode((400, 400))
    return win

def getKey(keyName):
    ans = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans

def main():
    win = init()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if getKey("LEFT"):
            print("Left key pressed")
        if getKey("RIGHT"):
            print("Right key pressed")
        if getKey("UP"):
            print("UP key pressed")
        if getKey("DOWN"):
            print("Down key pressed")
        if getKey("e"):
            print("E key pressed")
        if getKey("q"):
            print("Q key pressed")

if __name__ == '__main__':
    main()
