# phaser.py
# TODO: Needs to be cleaned up a lot

import numpy as np
from numpy.fft import fft, fftfreq, fftshift
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.path.insert(0, "../../chill/.packages/pyadi-iio/")  # Path to pyadi-iio library
import adi


def setup(phaser_config):
    my_sdr = adi.ad9361(uri=phaser_config["sdr_ip"])
    my_phaser = adi.CN0566(uri=phaser_config["rpi_ip"], rx_dev=my_sdr)

    # Initialize both ADAR1000s, set gains to max, and all phases to 0
    max_phaser_gain = int(2 ** int(phaser_config["max_gain_bits"]) - 1)
    my_phaser.configure(device_mode="rx")
    for i in range(0, 8):
        my_phaser.set_chan_gain(i, max_phaser_gain)
        my_phaser.set_chan_phase(i, 0)

    # ADAR1000 devices and channels for each
    num_devs = phaser_config["num_devs"]
    num_channels = phaser_config["num_channels"]

    sample_rate = float(phaser_config["sample_rate_mhz"]) * 1e6
    center_freq = float(phaser_config["center_freq_mhz"]) * 1e6
    signal_freq = float(phaser_config["signal_freq_mhz"]) * 1e6
    num_slices = int(phaser_config["num_slices"])
    fft_size = 1024 * 16

    # Create radio
    my_sdr.sample_rate = int(sample_rate)

    # Configure Rx
    my_sdr.rx_lo = int(center_freq)  # set this to output_freq - (the freq of the HB100)
    my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
    my_sdr.rx_buffer_size = int(fft_size)
    my_sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
    my_sdr.gain_control_mode_chan1 = "manual"  # manual or slow_attack
    my_sdr.rx_hardwaregain_chan0 = int(30)  # must be between -3 and 70
    my_sdr.rx_hardwaregain_chan1 = int(30)  # must be between -3 and 70

    # Configure Tx
    my_sdr.tx_lo = int(center_freq)
    my_sdr.tx_enabled_channels = [0, 1]
    my_sdr.tx_cyclic_buffer = True  # must set cyclic buffer to true for the tdd burst mode.  Otherwise Tx will turn on and off randomly
    my_sdr.tx_hardwaregain_chan0 = -88  # must be between 0 and -88
    my_sdr.tx_hardwaregain_chan1 = -0  # must be between 0 and -88

    # Enable TDD logic in pluto (this is for synchronizing Rx Buffer to ADF4159 TX input)
    # gpio = adi.one_bit_adc_dac(sdr_ip)
    # gpio.gpio_phaser_enable = True

    # Configure the ADF4159 Rampling PLL
    output_freq = 12.1e9
    c = 3e8
    wavelength = c / output_freq
    BW = 500e6
    num_steps = 1000
    ramp_time = 1e3  # us
    ramp_time_s = ramp_time / 1e6
    my_phaser.frequency = int(output_freq / 4)  # Output frequency divided by 4
    my_phaser.freq_dev_range = int(
        BW / 4
    )  # frequency deviation range in Hz.  This is the total freq deviation of the complete freq ramp
    my_phaser.freq_dev_step = int(
        BW / num_steps
    )  # frequency deviation step in Hz.  This is fDEV, in Hz.  Can be positive or negative
    my_phaser.freq_dev_time = int(
        ramp_time
    )  # total time (in us) of the complete frequency ramp
    my_phaser.delay_word = 4095  # 12 bit delay word.  4095*PFD = 40.95 us.  For sawtooth ramps, this is also the length of the Ramp_complete signal
    my_phaser.delay_clk = "PFD"  # can be 'PFD' or 'PFD*CLK1'
    my_phaser.delay_start_en = 0  # delay start
    my_phaser.ramp_delay_en = 0  # delay between ramps.
    my_phaser.trig_delay_en = 0  # triangle delay
    # Enable ADF4159 TX input and generate a single triangular ramp with each trigger
    my_phaser.ramp_mode = "continuous_triangular"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
    my_phaser.sing_ful_tri = 0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
    my_phaser.tx_trig_en = 0  # start a ramp with TXdata
    my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

    fs = int(my_sdr.sample_rate)
    ts = 1 / float(fs)
    N = int(my_sdr.rx_buffer_size)

    return my_sdr, my_phaser, N, ts


def tx(my_sdr, phaser_config):
    fs = int(my_sdr.sample_rate)
    ts = 1 / float(fs)
    N = int(my_sdr.rx_buffer_size)
    fc = int((float(phaser_config["signal_freq_mhz"]) * 1e6) / (fs / N)) * (fs / N)
    t = np.arange(0, N * ts, ts)

    i = np.cos(2 * np.pi * t * fc) * 2**14
    q = np.sin(2 * np.pi * t * fc) * 2**14
    iq = 1 * (i + 1j * q)

    my_sdr._ctx.set_timeout(60000)
    my_sdr.tx([iq * 0.5, iq])  # only send data to the 2nd channel (that's all we need)


def rx(my_sdr):
    N = int(my_sdr.rx_buffer_size)
    fs = int(my_sdr.sample_rate)
    ts = 1 / float(fs)

    signal = my_sdr.rx()
    signal = np.array(signal).sum(axis=0)

    signal_fft = np.abs(fft(signal))
    signal_fft /= signal_fft.max()
    freq = np.linspace(-fs / 2, fs / 2, N)
    # freq = fftfreq(N, d=ts)

    my_sdr.tx_destroy_buffer()

    return signal, signal_fft, freq
