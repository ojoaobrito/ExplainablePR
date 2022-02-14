############################################################################################################################
# API CLASS USED TO DRAW ONE OR MORE PLOTS IN THE SAME FIGURE (ON TOP OF EACH OTHER)
############################################################################################################################
import sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class Plot:

    def __init__(self, axis, title, x_label, y_label): # class constructor
        
        '''
        SHOULD BE CALLED LIKE THIS:

        variable_name = Plot([
            [ [0], [0], "-x", "r", "line_1" ], # an example plot (i.e. x_value, y_value, line_style, line_color, line_label)
            ...
            ], "title", "x_axis_label", "y_axis_label")
        '''
        
        plt.ion()
        self.axis = axis
        self.lines = []
        
        # initial configurations
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        # save the plot(s)
        for i in self.axis: self.lines.append(self.ax.plot(i[0], i[1], i[2], color = i[3], label = i[4]))
        
        # set the limits for each axis
        self.ax.set_xlim([min(list(map(lambda x : min(x[0]), self.axis))), max(list(map(lambda x : max(x[0]), self.axis)))])
        self.ax.set_ylim([min(list(map(lambda x : min(x[1]), self.axis))), max(list(map(lambda x : max(x[1]), self.axis)))])
        
        # add the title, labels and legend
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.legend()

        # show the initial plot(s)
        plt.grid()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, new_values): # update the plot(s)

        '''
        SHOULD BE CALLED LIKE THIS:

        variable_name.update([(plot1_new_x_value, plot1_new_y_val), ...])
        '''

        plt.ion()
        
        # add the new data to the axis
        for idx, i in enumerate(new_values):
            self.axis[idx][0].append(i[0])
            self.axis[idx][1].append(i[1])

        # update the plot(s)
        for idx, i in enumerate(self.axis): self.lines[idx][0].set_data(i[0], i[1])
        
        # set the limits for each axis
        self.ax.set_xlim([min(list(map(lambda x : min(x[0]),self.axis))),max(list(map(lambda x : max(x[0]),self.axis)))])
        self.ax.set_ylim([min(list(map(lambda x : min(x[1]),self.axis))),max(list(map(lambda x : max(x[1]),self.axis)))])

        # show the new plot(s) and wait some time
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def keep(self): # keep the plot(s) on the screen
        
        '''
        SHOULD BE CALLED LIKE THIS:

        variable_name.keep()
        '''

        plt.ioff()
        plt.show()

    def save(self, image_name): # save the plot(s) (as an image)
        
        '''
        SHOULD BE CALLED LIKE THIS:

        variable_name.save(image_name)
        '''

        plt.savefig(image_name)

if(__name__ == "__main__"):
    
    ######################################################################################################
    # EXAMPLE EXECUTION
    ######################################################################################################
    # create a "Plot" object
    plot = Plot([
        [ [0], [0], "-x", "r", "line_1" ], # line number 1 (in red)
        [ [0], [0], "-x", "g", "line_2" ]  # line number 2 (in green)
        ],"TITLE", "X AXIS", "Y AXIS")

    # update the plot(s)
    for i in range(15): 
        plot.update([
            (i + 1, i * 2), # new value for plot 1
            (i + 1, i * 2.5) # new value for plot 2
            ])

    # save the plot(s) - NOTE: never call this method after "keep()", otherwise the plot(s) will disappear
    plot.save("test.png")

    # prevent the plot(s) from going away
    plot.keep()