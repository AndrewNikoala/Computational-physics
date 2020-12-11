import PySimpleGUI as sg 
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import explicitScheme2D

# set theme for the window-interface
sg.theme('LightGrey2') 

# general settings
font = ("Helvetica", 15)

# define layout
layout = \
[
    [sg.Text('Explicit scheme', font=font)],
    
    [sg.Text('Gif of solution:', font=font),
    sg.Button('Gif', key='explGif', font=font)],
    
    [sg.Text('Draw plot of the central Temperatures:', font=font),
    sg.Button('Plot', key='explPlot', font=font)],
    
    [sg.Text('Adjusting parameters', font=font)],
    
    [sg.Text('Number of X nodes', font=font),
    sg.Input('10', key='-INX-', font=font, size = (5, 1), enable_events=True)],
    [sg.Text('Number of Y nodes', font=font),
    sg.Input('10', key='-INY-', font=font, size = (5, 1), enable_events=True)],
    [sg.Text('Number of t nodes ', font=font),
    sg.Input('100', key='-INt-', font=font, size = (5, 1), enable_events=True)],
    [sg.Button('Set parameters', key='setPar', font=font)],
    
    [sg.Text('Save Gif:', font=font),
    sg.Input('HeatingEquation', key='-FileName-', size = (20, 1), font=font, enable_events=True),
    sg.Button('Save gif', key='saveGif', font=font)],
]

# Create interface-window
window = sg.Window('Heating equation', layout, finalize=True, resizable=True, size=(500, 350))

# Init

Sol = explicitScheme2D.SolutionGrid(10, 10, 100)

# handling of events
while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'explGif':
        Sol.drawGifSolution()
        
    if event == 'explPlot':
        Sol.drawCentreTemp()
        
    if event == '-INX-':
        window['-INY-'].update(values['-INX-'])
        
    if event == '-INY-':
        window['-INX-'].update(values['-INY-'])
        
    if event == 'setPar':
        if values['-INt-'] == '':
            values['-INt-'] = 1
            window['-INt-'].update(values['-INt-'])
        
        if values['-INX-'] == '' or values['-INY-'] == '' or int(values['-INX-']) < 3 or int(values['-INY-']) < 3:
            values['-INX-'] = 3
            values['-INY-'] = 3
            window['-INX-'].update(values['-INX-'])
            window['-INY-'].update(values['-INY-'])
            
        Sol = explicitScheme2D.SolutionGrid(int(values['-INX-']), int(values['-INY-']), int(values['-INt-']))
    
    if event == 'saveGif':
        Sol.animationSave(values['-FileName-'] + '.gif')
        
window.close()
