import numpy as np



def _s_transform_core(
        signal,
        fs,
        adaptive=False,
        integration_window=None):

    """
    Time-localized S-transform

    S(tau,f)=sum x(t) w(t-tau,f) exp(-j2pi ft)

    integration_window:
        seconds
    """

    N=len(signal)


    dt=1/fs


    # frequency axis

    freq=np.fft.rfftfreq(
        N,
        dt
    )


    spectrum=np.fft.fft(
        signal
    )



    # time axis

    time=np.arange(N)/fs



    ST=[]



    for f in freq:


        # adaptive Gaussian width

        if adaptive:

            sigma = 0.5/(abs(f)+1e-12)

        else:

            sigma = 1.0/(abs(f)+1e-12)



        # integration interval

        if integration_window is not None:


            half = integration_window/2


            window_mask=(

                np.abs(

                    time-time[N//2]

                )

                <= half

            )


        else:

            window_mask=np.ones(
                N,
                dtype=bool
            )



        gaussian=np.exp(

            -0.5*

            (

                (time-time[N//2])

                /

                (sigma+1e-12)

            )**2

        )



        gaussian *= window_mask



        gaussian /= (

            np.sum(gaussian)

            +

            1e-12

        )



        # localized Fourier component

        component=np.sum(

            signal *

            gaussian *

            np.exp(

                -1j*

                2*np.pi*f*time

            )

        )


        ST.append(
            component
        )



    ST=np.array(ST)



    return freq,ST




def discrete_s_transform(
        signal,
        fs,
        integration_window=None):


    return _s_transform_core(

        signal,

        fs,

        adaptive=False,

        integration_window=

        integration_window

    )





def adaptive_s_transform(
        signal,
        fs,
        integration_window=None):


    return _s_transform_core(

        signal,

        fs,

        adaptive=True,

        integration_window=

        integration_window

    )

