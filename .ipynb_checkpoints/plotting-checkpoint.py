def quadplot( title, plot00, plot10, plot10, plot11, xlabels, ylabels):
    mpl.rcParams['figure.figsize'] = [15.0, 6.0]
    f, ax_arr = plt.subplots(2, 2)
    f.suptitle(title)
    if plot00ax_arr[0,0].( plot00[0,0], plot00[0,1], c=plot00[1,0], label=plot00[1,1], ls=plot00[1,2])
    ax_arr[0,0].errorbar( times, phi1_fitresults_arr[i], (phi1_err[1]+phi1_err[0]*times), c="red", ls="--", label="Fit")
    ax_arr[1,0].errorbar( times, (phi1_arr_all[i] - phi1_fitresults_arr[i]), phi1_err_all[i], c="green", label="Res.")
    ax_arr[1,0].axhline(0, ls="--", c="grey")
    ax_arr[0,0].legend(frameon=False, ncol=2 )
    ax_arr[1,0].legend(frameon=False, ncol=2 )
    ax_arr[1,0].set_xlabel("Time [sec]")
    ax_arr[0,0].set_ylabel("Phase [rad]")
    ax_arr[1,0].set_ylabel("Residuals")
    f.subplots_adjust(hspace=0)