import math
from tkinter import *
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
from comparesignals import SignalSamplesAreEqual

t1 = []
y1 = []
t2 = []
y2 = []
y3 = []

def compare_browse():
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
#    expectedOutput.set(filename)
    SignalSamplesAreEqual(filename,t1,y3)

def signal(Amp, AF, Theta, t_signal):  # Function to calc sin or cos
    if (clicked1.get() == "sin"):
        y = Amp * np.sin((2 * np.pi * AF * t_signal) + Theta)
    elif (clicked1.get() == "cos"):
        y = Amp * np.cos((2 * np.pi * AF * t_signal) + Theta)
    return y

def plot():  # generate the signal
    Amp = float(inputAmp.get(1.0, "end"))
    AF = float(inputAF.get(1.0, "end"))  # Analog Frequency
    SF = float(inputSF.get(1.0, "end"))  # Sampling Frequency
    Theta = float(inputTheta.get(1.0, "end"))

    fig, axs = plt.subplots(2)

    if SF != 0:
        if SF < 2 * AF:
            labeltext.set("set Sample Frequency to bigger than 2 x Analog Frequency\n or set Sample Frequency to 0 for continous signal")
            return
        labeltext.set("Run")
        t_sampled = np.arange(0, 6 / AF, 1 / SF)  # Sampling time instants
        y_sampled = signal(Amp, AF, Theta, t_sampled)
        axs[1].stem(t_sampled, y_sampled)
        axs[1].set_ylim([-Amp, Amp])

    t = np.arange(0, 5 / AF, 0.00001)  # Sampling time instants
    y = signal(Amp, AF, Theta, t)
    axs[0].plot(t, y)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=20, column=0, columnspan=20)

def plot_s1():
    fig, axs = plt.subplots(2)
    axs[0].plot(t1, y1)
    axs[1].stem(t1, y1)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=20, column=0, columnspan=20)

def plot_s2():
    fig, axs = plt.subplots(2)
    axs[0].plot(t2, y2)
    axs[1].stem(t2, y2)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=20, column=0, columnspan=20)

def browse1():
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    s1_name.set(filename)

    with open(filename) as f:
        lines = f.readlines()

    t1[:], y1[:] = [], []
    for line in lines[3:]:
        x, y, z = line.split(" ")
        t1.append(float(x))
        y1.append(float(y))

def browse2():
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    s2_name.set(filename)

    with open(filename) as f:
        lines = f.readlines()

    t2[:], y2[:] = [], []
    for line in lines[3:]:
        x, y, z = line.split(" ")
        t2.append(float(x))
        y2.append(float(y))


def Add():
    y3 = np.add(y1,y2)
    fig, axs = plt.subplots(3)
    axs[0].plot(t1, y1)
    axs[1].plot(t2, y2)
    axs[2].plot(t1, y3)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=20)


def Subtract():
    y3 = np.subtract(y1,y2)
    fig, axs = plt.subplots(3)
    axs[0].plot(t1, y1)
    axs[1].plot(t2, y2)
    axs[2].plot(t1, y3)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,master=root)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=15)

def Multiply():
    c = float(input_Multiply.get(1.0, "end"))
    y3 = [y * c for y in y1]
    fig, axs = plt.subplots(2)
    axs[0].plot(t1, y1)
    axs[1].plot(t1, y3)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=15)

def Square():
    y3 = [y * y for y in y1]
    fig, axs = plt.subplots(2)
    axs[0].plot(t1, y1)
    axs[1].plot(t1, y3)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=15)

def Normalize():
    y_min = min(y1)
    y_max = max(y1)
    a = float(input_min.get(1.0, "end"))
    b = float(input_max.get(1.0, "end"))

    y3 = [((b - a) * (y - y_min) / (y_max - y_min)) + a for y in y1]
    fig, axs = plt.subplots(2)
    axs[0].plot(t1, y1)
    axs[1].plot(t1, y3)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,master=root)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=15)

def Accumulate():
    #y3 = np.cumsum(y1)
    accumulated = []
    total = 0
    for value in y1:
         total += value
         accumulated.append(total)

    y3=accumulated
    fig, axs = plt.subplots(2)
    axs[0].plot(t1, y1)
    axs[1].plot(t1, y3)
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=7, column=0, columnspan=15)
    
############# task 33333333333333333333333333   ##########################    

def Quantize():
    levels =None
    bits=None
    if(clicked2.get()=="levels"):
        levels=input_shit.get(1.0, "end")
    elif(clicked2.get()=="bits"):
        bits=input_shit.get(1.0, "end")
        
    if levels is None and bits is not None:
        levels = 2 ** int(bits)
    elif levels is not None:
        bits = math.ceil(math.log2(int(levels)))
    else:
        raise ValueError("Either levels or bits must be provided")
    intervals, midpoints, bits = generate_intervals_and_midpoints(y1, levels=levels, bits=bits)
    original_numbers, encoded_binaries, interval_midpoints,error_numbers,indexNumber = assign_encoded_intervals(y1, intervals, midpoints, bits)

    # Convert midpoints to regular floats for cleaner output
    interval_midpoints = [round(float(midpoint),3) for midpoint in interval_midpoints]

    # Print results
    print("Numbers:", original_numbers)
    print("index: ",indexNumber)
    print("Encoded Binaries:", encoded_binaries)
    print("Midpoints:", interval_midpoints)
    print("error: ",error_numbers)
    return
def generate_intervals_and_midpoints(array, levels=None, bits=None):
    # Calculate min and max of the array
    min_value = np.min(array)
    max_value = np.max(array)

    # Determine the number of levels based on either bits or levels
    if levels is None and bits is not None:
        levels = 2 ** bits
    elif levels is not None:
        bits = math.ceil(math.log2(levels))
    else:
        raise ValueError("Either levels or bits must be provided")

    # Calculate delta based on the number of levels
    delta = (max_value - min_value) / levels

    # Create intervals and midpoints lists
    intervals = []
    midpoints = []

    # Generate intervals and their midpoints, covering the full range
    start = min_value
    while start < max_value:
        end = start + delta
        intervals.append((start, end))
        midpoints.append((start + end) / 2)
        start = end
    midpoints=[round(float(midpoint),3) for midpoint in midpoints]
    float_matrix = [[round(float(value),3) for value in row] for row in intervals]
    #print(float_matrix,midpoints)
    return float_matrix, midpoints, bits

def encode_index(index, bits):
    return format(index, f'0{bits}b')

def assign_encoded_intervals(numbers, intervals, midpoints, bits):
    # Separate lists for the output
    encoded_binaries = []
    interval_midpoints = []
    original_numbers = []
    error_numbers=[]
    index_number=[]

    for number in numbers:
        # Find the interval for each number
        for i, (start, end) in enumerate(intervals):
            if start <= number <= end:
                # Assign values to separate lists
                encoded_binaries.append(encode_index(i, bits))
                interval_midpoints.append(midpoints[i])
                original_numbers.append(number)
                error_numbers.append(round(midpoints[i]-number,3))
                index_number.append(i+1)
                break
        else:
            # If the number is exactly the max value, put it in the last interval
            if number == np.max(numbers):
                encoded_binaries.append(encode_index(len(intervals) - 1, bits))
                interval_midpoints.append(midpoints[-1])
                original_numbers.append(number)
                error_numbers.append(round(midpoints[i]-number,3))
                index_number.append(i+1)

    return original_numbers, encoded_binaries, interval_midpoints,error_numbers,index_number

############## TASK 44444444444444444444444444 ##########################
def custom_idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        sum_val = sum(X[k] * np.exp(2j * np.pi * k * n / N) for k in range(N))
        x[n] = sum_val / N
    return x

# Function to load and process file data
def load_signal_file():
    file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=(("Text Files", "*.txt"),))
    if not file_path:
        return None

    with open(file_path, 'r') as file:
        lines = file.readlines()

    domain_type = int(lines[1].strip())

    if domain_type == 0:
        time_indices = []
        output_values = []
        for line in lines[3:]:
            if len(line.strip().split()) != 2:
                continue
            index, value = map(float, line.strip().split())
            time_indices.append(index)
            output_values.append(value)
        return domain_type, np.array(time_indices), np.array(output_values)

    elif domain_type == 1:
        amplitude = []
        phase = []
        for line in lines[3:]:
            if ',' not in line:
                continue
            freq_amp, phase_value = line.replace('f', '').split(',')
            amplitude.append(float(freq_amp.strip()))
            phase.append(float(phase_value.strip()))
        return domain_type, np.array(amplitude), np.array(phase)

    else:
        print("Error: Invalid domain type in file.")
        return None

# Function to perform Fourier Transform and display results
# def apply_fourier_transform():
#     sampling_frequency = simpledialog.askfloat("Input", "Enter sampling frequency (Hz):")
#     if not sampling_frequency:
#         return
#
#     domain_type, time_indices, signal_values = load_signal_file()
#     if domain_type == 0:
#         fft_values = np.fft.fft(signal_values)
#         freqs = np.fft.fftfreq(len(signal_values), 1 / sampling_frequency)
#
#         amplitude = np.abs(fft_values)
#         phase = np.angle(fft_values)
#         amplitude_list = ["{:.2f}".format(amp) for amp in amplitude]
#         phase_list = ["{:.2f}".format(ph) for ph in phase]
#
#         print("Amplitude List:", amplitude_list)
#         print("Phase List:", phase_list)
#
#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.stem(freqs, amplitude, label="Amplitude", linefmt="blue", markerfmt="bo", basefmt=" ")
#         plt.xlabel("Frequency (Hz)")
#         plt.ylabel("Amplitude")
#         plt.title("Frequency vs Amplitude")
#         plt.grid()
#         plt.legend()
#
#         plt.subplot(1, 2, 2)
#         plt.stem(freqs, phase, label="Phase", linefmt="orange", markerfmt="ro", basefmt=" ")
#         plt.xlabel("Frequency (Hz)")
#         plt.ylabel("Phase (radians)")
#         plt.title("Frequency vs Phase")
#         plt.grid()
#         plt.legend()
#
#         plt.tight_layout()
#         plt.show()
#
#     else:
#        messagebox.showinfo("Info", "The signal is already in the frequency domain.")


def compute_dft(samples, fs):
    """
    Compute the Discrete Fourier Transform (DFT) of a signal.

    Args:
        samples (list): Time-domain signal samples.
        fs (float): Sampling frequency in Hz.

    Returns:
        tuple: (frequencies, magnitudes, phases)
               - frequencies: List of frequency indices.
               - magnitudes: List of amplitude magnitudes.
               - phases: List of phase angles (in radians).
    """
    N = len(samples)  # Number of samples
    magnitudes = []
    phases = []

    # DFT Calculation
    for k in range(N):
        real_part = 0
        imag_part = 0
        for n in range(N):
            exponent = (2 * math.pi * k * n) / N
            real_part += samples[n] * math.cos(exponent)
            imag_part -= samples[n] * math.sin(exponent)

        magnitudes.append(math.sqrt(real_part ** 2 + imag_part ** 2))  # Amplitude
        phases.append(math.atan2(imag_part, real_part))  # Phase

    # Frequency indices
    freq_step = fs / N
    frequencies = [k * freq_step for k in range(N)]

    return frequencies, magnitudes, phases


def apply_fourier_transform():
    # Prompt user for sampling frequency
    sampling_frequency = simpledialog.askfloat("Input", "Enter sampling frequency (Hz):")
    if not sampling_frequency:
        return

    # Load signal data (domain type, time indices, and signal values)
    domain_type, time_indices, signal_values = load_signal_file()

    # If the signal is in time domain (domain_type == 0)
    if domain_type == 0:
        # Compute the DFT
        frequencies, magnitudes, phases = compute_dft(signal_values, sampling_frequency)

        # Print and plot the results
        print("Frequencies:", frequencies)
        print("Magnitudes:", magnitudes)
        print("Phases:", phases)

        # Plot the DFT Amplitude and Phase Spectrum
        plt.figure(figsize=(10, 8))

        # Amplitude Spectrum
        plt.subplot(2, 1, 1)  # First subplot (top)
        plt.stem(frequencies, magnitudes, linefmt="b-", markerfmt="bo", basefmt="r-")
        plt.title("DFT Amplitude Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()

        # Phase Spectrum
        plt.subplot(2, 1, 2)  # Second subplot (bottom)
        plt.stem(frequencies, phases, linefmt="b-", markerfmt="bo", basefmt="r-")
        plt.title("DFT Phase Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (radians)")
        plt.grid()

        # Tight layout for clarity
        plt.tight_layout()
        plt.show()

    else:
        messagebox.showinfo("Info", "The signal is already in the frequency domain.")
# Function to perform Inverse Discrete Fourier Transform and reconstruct signal
def reconstruct_signal_from_frequency_domain():
    domain_type, amplitude, phase = load_signal_file()
    if domain_type != 1:
        messagebox.showinfo("Info", "The signal is not in the frequency domain.")
        return

    # Create complex frequency values using amplitude and phase
    signal_values = amplitude * np.exp(1j * phase)

    # Perform inverse Fourier transform
    reconstructed_signal = custom_idft(signal_values)

    time_indices = np.arange(len(reconstructed_signal))
    output_list = []
    for time, value in zip(time_indices, np.real(reconstructed_signal)):
        output_list.append(f"Time: {time}, Amplitude: {value:.2f}")

    print("\nFormatted Output List:")
    for output in output_list:
        print(output)   
        
# Plot the reconstructed signal with discrete markers
    plt.figure()
    plt.stem(time_indices, np.real(reconstructed_signal), linefmt="blue", markerfmt="bo", basefmt=" ", label="Reconstructed Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Reconstructed Signal in Time Domain")
    plt.grid()
    plt.legend()
    plt.show()


'''_____________________ GUI _____________________ '''
# Create object
root = Tk()

# Adjust size
root.geometry("1000x700")

'''__________________________Task11111111111111111_____________________'''

# Dropdown menu options
options = [ "sin", "cos" ]

# datatype of menu text
clicked1 = StringVar()

# initial menu text
clicked1.set("sin")

# Create Dropdown menu
drop = OptionMenu(root, clicked1, *options)
drop.grid(row=0, column=0)

'''----------------------------Amplitude--------------------------'''
Amp_lbl = Label(root, text="Amplitude")
Amp_lbl.grid(row=1, column=0)
# TextBox Creation
inputAmp = Text(root, height=1, width=5)
inputAmp.grid(row=1, column=1)

'''----------------------------Theta--------------------------'''
Theta_lbl = Label(root, text="Theta")
Theta_lbl.grid(row=2, column=0)
# TextBox Creation
inputTheta = Text(root, height=1, width=5)
inputTheta.grid(row=2, column=1)

'''----------------------------AnalogFrequency-------------------------'''
AF_lbl = Label(root, text="AnalogFrequency")
AF_lbl.grid(row=3, column=0)
# TextBox Creation
inputAF = Text(root, height=1,  width=5)
inputAF.grid(row=3, column=1)

'''----------------------------SamplingFrequency----------------------'''
SF_lbl = Label(root, text="SamplingFrequency")
SF_lbl.grid(row=4, column=0)
# TextBox Creation
inputSF = Text(root, height=1, width=5)
inputSF.grid(row=4, column=1)

# Create button, it will change label text
button = Button(root, text="Show the Signal", command=plot).grid(row=5, column=0)

'''__________________________Task2222222222222_____________________'''
# Signal text
s1_name = StringVar()
s1_name.set("s1")

# Create Label
s1_label = Label(root, textvariable=s1_name)
s1_label.grid(row=0, column=2)

# Create button, it will change label text
s1_button = Button(root, text="Choose Signal1", command=browse1).grid(row=1, column=2)

# Create button, it will change label text
button1 = Button(root, text="Plot Signal1", command=plot_s1).grid(row=2, column=2)

# Signal text
s2_name = StringVar()
s2_name.set("s2 ")

# Create Label
s2_label = Label(root, textvariable=s2_name)
s2_label.grid(row=3, column=2)

# Create button, it will change label text
choose_Signal_1_button = Button(root, text="Choose Signal 2", command=browse2).grid(row=4, column=2)
# Create button, it will change label text
button_Plot_Signal_1 = Button( root , text = "Plot Signal 2" , command = plot_s2 ).grid(row=5,column=2)

# Create button, it will change label text
compare_button = Button(root, text="Compare Output", command= compare_browse).grid(row=4, column=7)

# Create button, it will change label text
Add_button = Button( root , text = "Add" , command = Add ).grid(row=0,column=4)
# Create button, it will change label text
subtract_button = Button( root , text = "Subtract" , command = Subtract ).grid(row=1,column=4)
# Create button, it will change label text
multiply_button = Button( root , text = "Multiply" , command = Multiply ).grid(row=2,column=4)
# TextBox Creation
input_Multiply = Text(root, height = 1, width = 5)
input_Multiply.grid(row=2,column=5)
# Create button, it will change label text
square_button = Button( root , text = "Square" , command = Square ).grid(row=3,column=4)
# Create button, it will change label text
Normalize_button = Button( root , text = "Normalize" , command = Normalize ).grid(row=4,column=4)
# TextBox Creation
input_min = Text(root, height=1, width=5)
input_min.grid(row=4,column=5)
# TextBox Creation
input_max = Text(root, height=1, width=5)
input_max.grid(row=4,column=6)
# Create button, it will change label text
Accumulate_button = Button(root, text="Accumulate", command=Accumulate).grid(row=5,column=4)
'''                  TASK 3                '''
options = [ "levels", "bits" ]

# datatype of menu text
clicked2 = StringVar()

# initial menu text
clicked2.set("levels")

# Create Dropdown menu
drop = OptionMenu(root, clicked2, *options)
drop.grid(row=0, column=8)

# Signal text
s1_name = StringVar()
s1_name.set("s1")

# Create Label
s1_label = Label(root, textvariable=s1_name)
s1_label.grid(row=1, column=8)

# Create button, it will change label text
s1_button = Button(root, text=" Signal to Q", command=browse1).grid(row=2, column=8)

# Create button, it will change label text
Q_button = Button( root , text = "Quantize" , command = Quantize).grid(row=4,column=8)
# TextBox Creation
input_shit = Text(root, height = 1, width = 5)
input_shit.grid(row=3,column=8)

'''                  Task 4                 '''

menu = Menu(root)
root.config(menu=menu)
freq_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Frequency Domain", menu=freq_menu)
freq_menu.add_command(label="Apply Fourier Transform", command=apply_fourier_transform)
freq_menu.add_command(label="Reconstruct Signal (IDFT)", command=reconstruct_signal_from_frequency_domain)


labeltext = StringVar()
labeltext.set(" ")

# Create Label
label = Label(root, textvariable=labeltext)
label.grid(row=5, column=1)

# Execute tkinter
root.mainloop()
