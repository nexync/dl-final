import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.ion()  # Turn on interactive mode

def make_plot(scores, mean_scores, fig, ax):
    # print('scores:', scores)
    # print('mean_scores:', mean_scores)
    ax.clear()  # Clear previous data
    ax.plot(scores, 'o-', label='Scores')  
    ax.plot(mean_scores, 's-', label='Mean Scores') 
    ax.set_title('Training...')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Score')
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left')
    if scores:
        ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.draw()  # Redraw the plot
    plt.pause(0.1)  
    plt.show(block=False)
    # Optionally save the figure
    fig.savefig('training_plot.png')