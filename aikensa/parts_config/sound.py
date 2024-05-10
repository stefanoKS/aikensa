import pygame
import time

keisoku_sound = pygame.mixer.Sound("aikensa/sound/mixkit-bell-notification-933.wav") 
konpou_sound = pygame.mixer.Sound("aikensa/sound/mixkit-software-interface-back-2575.wav")



def play_keisoku_sound():
    keisoku_sound.play()

def play_konpou_sound():
    konpou_sound.play()
