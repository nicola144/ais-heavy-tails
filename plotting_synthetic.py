import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)

plt.rc('font', family='serif', size=18)
plt.rc('axes', labelsize=25, titlesize=20, labelpad=20)  # Adjusting axes parameters
plt.rc('xtick', labelsize=20)  # Adjusting xtick parameters
plt.rc('ytick', labelsize=20)  # Adjusting ytick parameters
plt.rc('legend', handlelength=2)  # e.g., shorter lines
plt.rc('lines', markersize=8)  # Replace 10 with your desired size

# default_figsize = plt.rcParams['figure.figsize']

# Function to manage colors per marker type
def get_color_for_marker(marker, marker_to_color, color_cycle):
    if marker not in marker_to_color:
        try:
            new_color = next(color_cycle)
        except StopIteration:
            # Restart the color cycle if it has ended
            color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            new_color = next(color_cycle)
        marker_to_color[marker] = new_color
    return marker_to_color[marker]


# Prepare the color cycle and marker-color mapping
color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
marker_to_color = {}



d_collect = [2,4,8,16]


# target dof equal to 2
true_Z = np.array([1.404962946208144459e+01, 9.869604401089360124e+01, 1.623484850566705973e+03, 3.765290085742292285e+04])
targetName = "dofTarget2"

alphaESS_mean_escortAMIS_dof1 = np.array([9.144814464849421176e-01, 8.981825945966060987e-01, 8.880906695810214524e-01, 8.826532506289761493e-01])
alphaESS_mean_escortAMIS_dof2 = np.array([9.999913080593623160e-01, 9.999752324964497063e-01, 9.999035903480074650e-01, 9.996008552874264819e-01])
alphaESS_mean_escortAMIS_dof3 = np.array([9.703107581569057061e-01, 9.588732348477198020e-01, 9.521499686254339956e-01, 9.473129140156296701e-01])
alphaESS_mean_escortAMIS_dof5 = np.array([8.702389160398555923e-01, 8.337632079177429434e-01, 7.820506228753225653e-01, 7.626158036003678475e-01])
alphaESS_mean_escortAMIS_dof10 = np.array([7.825653887515278129e-01, 6.958181386442454652e-01, 5.805113123272713782e-01, 5.076722166787422941e-01])

alphaESS_std_escortAMIS_dof1 = np.array([1.876491026996742612e-03, 1.814864203198255331e-03, 2.034501142463264901e-03, 1.876746922200428101e-03])
alphaESS_std_escortAMIS_dof2 = np.array([6.136402240386612833e-06, 9.787650853634096305e-06, 2.021958748877326046e-05, 4.507892638231119129e-05])
alphaESS_std_escortAMIS_dof3 = np.array([4.830985918140424246e-03, 7.246578048611529779e-03, 4.862451845208858904e-03, 4.625628624005758208e-03])
alphaESS_std_escortAMIS_dof5 = np.array([1.041486667041957304e-01, 4.374593000605125209e-02, 4.589682622705657661e-02, 4.587162275060568950e-02])
alphaESS_std_escortAMIS_dof10 = np.array([1.400719423505334404e-01, 1.007743183913678009e-01, 1.141211380411730758e-01, 1.053985939329236815e-01])

MSE_Z_escortAMIS_dof1 = np.array([1.155109970719065895e-04, 8.328016828798572024e-03, 2.827138705627340265e+00, 3.060072055903096953e+03])
MSE_Z_escortAMIS_dof2 = np.array([1.283719866724547356e-05, 5.155852660450637370e-04, 3.728151849836758847e-01, 1.263489762824257241e+03])
MSE_Z_escortAMIS_dof3 = np.array([5.810489344318569757e-05, 3.759686412102727741e-03, 1.959848786973888091e+00, 2.301547826966623688e+03])
MSE_Z_escortAMIS_dof5 = np.array([8.638888066407172679e-04, 6.519081092903490637e-02, 3.192798568028252859e+01, 4.802412997816570714e+04])
MSE_Z_escortAMIS_dof10 = np.array([7.490752351310303592e-03, 9.753313423075428146e-01, 7.459457109121632357e+02, 1.477757177243437385e+06])


alphaESS_mean_adaptive = np.array([9.923895884629682085e-01, 9.870077633517324589e-01, 9.920454581998857657e-01, 9.910579921332822728e-01])
alphaESS_std_adaptive = np.array([2.344605314529271331e-02, 4.910411401587817110e-02, 3.810206097003925041e-02, 4.898319924375282852e-02])
MSE_Z_adaptive = np.array([1.366206052759713757e-05, 9.263725499640704459e-04, 5.740420487947989203e-01, 1.558374659628888821e+03])


alphaESS_mean_AMIS_dof3 = np.array([7.659374575652233563e-01, 6.664563544084158853e-01, 5.786582472080282757e-01, 5.797713148235533742e-01])
alphaESS_mean_AMIS_dof5 = np.array([7.015030943837191613e-01, 5.256082349212668481e-01, 3.818471395949649683e-01, 3.048852071993036206e-01])
alphaESS_mean_AMIS_dof10 = np.array([7.009875561134312205e-01, 5.478054410991100021e-01, 3.519177320376049023e-01, 2.522118632814392414e-01])

alphaESS_std_AMIS_dof3 = np.array([6.558522800371312245e-02, 5.021573768778590102e-02, 7.060628607807720170e-02, 7.351717874973173750e-02])
alphaESS_std_AMIS_dof5 = np.array([4.185772994488935056e-02, 5.710959651686075417e-02, 4.489901576815884254e-02, 8.204959792239004124e-02])
alphaESS_std_AMIS_dof10 = np.array([7.838414445943911457e-02, 6.444046138159498227e-02, 7.817927393964584082e-02, 8.422124797246401118e-02])

MSE_Z_AMIS_dof3 = np.array([4.468962243240218266e-04, 5.629800319176823248e-02, 2.215160901061101484e+01, 3.007291938423297688e+04])
MSE_Z_AMIS_dof5 = np.array([6.829300353220114786e-04, 1.428253062657661721e-01, 1.354714861697121364e+02, 5.795708411302637542e+05])
MSE_Z_AMIS_dof10 = np.array([5.241437234267612086e-03, 8.462517690205006993e-01, 6.275423370822865081e+02, 2.279684348320381250e+06])



# target dof equal to 5
# true_Z = np.array([1.409943485869908386e+02, 1.409943485869908386e+02, 7.028072946176218466e+03, 2.717306491258586291e+06])
# targetName = "dofTarget5"

# alphaESS_mean_escortAMIS_dof1 = np.array([7.934226127952928032e-01, 7.310923811449652199e-01, 6.816655391996948588e-01, 6.489872146801646258e-01])
# alphaESS_mean_escortAMIS_dof2 = np.array([9.392924600718375316e-01, 9.098510660575870324e-01, 8.815974452894139723e-01, 8.600488035754887406e-01])
# alphaESS_mean_escortAMIS_dof3 = np.array([9.833134463029642447e-01, 9.730357572222129159e-01, 9.617637198980206747e-01, 9.520765087739251342e-01])
# alphaESS_mean_escortAMIS_dof5 = np.array([9.999903890008293716e-01, 9.999699781629713957e-01, 9.998944452362458213e-01, 9.995926554995425706e-01])
# alphaESS_mean_escortAMIS_dof10 = np.array([9.821249630500021865e-01, 9.668297702504521363e-01, 9.368140838818128646e-01, 9.074190736738451113e-01])

# alphaESS_std_escortAMIS_dof1 = np.array([2.806880437443453735e-03, 3.347018290721677850e-03, 3.511392104322683637e-03, 3.570272544710990058e-03])
# alphaESS_std_escortAMIS_dof2 = np.array([1.599987824125829409e-03, 1.903984435380082088e-03, 2.055018412790969411e-03, 2.335316776825327630e-03])
# alphaESS_std_escortAMIS_dof3 = np.array([5.763685637842146884e-04, 9.240119711780532064e-04, 1.013819946176315037e-03, 9.813289938449770797e-04])
# alphaESS_std_escortAMIS_dof5 = np.array([6.513296917554514887e-06, 1.093678566449271963e-05, 1.940672622523351067e-05, 4.414081810184986249e-05])
# alphaESS_std_escortAMIS_dof10 = np.array([6.862363906509987531e-03, 1.025794507587121304e-02, 4.320897541740194997e-02, 3.025245245002826378e-02])

# MSE_Z_escortAMIS_dof1 = np.array([5.331004458700579988e-04, 4.813709540803217529e-02, 1.104990301540095743e+02, 4.597810320877866447e+07])
# MSE_Z_escortAMIS_dof2 = np.array([7.869276789534255276e-05, 1.544162448249263020e-02, 4.249017775722693102e+01, 1.965758856025061384e+07])
# MSE_Z_escortAMIS_dof3 = np.array([3.999427501772853091e-05, 6.293553464937198458e-03, 2.058921455179503113e+01, 1.009652632730942592e+07])
# MSE_Z_escortAMIS_dof5 = np.array([1.312796543613250550e-05, 1.378591416159999014e-03, 7.934710989683143367e+00, 6.285406252703314647e+06])
# MSE_Z_escortAMIS_dof10 = np.array([3.893314055018779558e-05, 7.287334515395931141e-03, 4.943531595714093640e+01, 2.765861453500132263e+07])


# alphaESS_mean_adaptive = np.array([9.999006891850575895e-01, 9.998496141254210601e-01, 9.993099756738911266e-01, 9.915802285376412728e-01])
# alphaESS_std_adaptive = np.array([2.186818811641501662e-04, 1.896168337427528786e-04, 1.155923309316938055e-03, 3.593198467896641796e-02])
# MSE_Z_adaptive = np.array([2.019963716080580004e-05, 1.312256467378856246e-03, 7.806592116210620880e+00, 7.184603819826675579e+06])


# alphaESS_mean_AMIS_dof3 = np.array([9.267350999215879792e-01, 8.839430186217240859e-01, 8.406173903007153614e-01, 8.053579847616778586e-01])
# alphaESS_mean_AMIS_dof5 = np.array([9.999739357129183714e-01, 9.999254711304466081e-01, 9.997479123107883003e-01, 9.990422163132122169e-01])
# alphaESS_mean_AMIS_dof10 = np.array([9.780486435922012278e-01, 9.544831835554397070e-01, 9.098606060494455194e-01, 8.569377252709959869e-01])

# alphaESS_std_AMIS_dof3 = np.array([8.828580536520690440e-04, 1.491191707571309488e-03, 2.172356185066151926e-03, 2.695195898399960818e-03])
# alphaESS_std_AMIS_dof5 = np.array([1.859939569304494241e-05, 3.442251978639737864e-05, 6.683963635418733648e-05, 1.369821477404313634e-04])
# alphaESS_std_AMIS_dof10 = np.array([5.461498387203026113e-03, 6.514856450486256230e-03, 1.369703667479143551e-02, 1.309882876782711078e-02])

# MSE_Z_AMIS_dof3 = np.array([1.169571818936132612e-04, 1.874001588714309732e-02, 8.796354567525187917e+01, 2.322256702357692271e+07])
# MSE_Z_AMIS_dof5 = np.array([9.360621769185713969e-06, 1.387316748219458162e-03, 4.683516993399246431e+00, 5.815979157076846808e+06])
# MSE_Z_AMIS_dof10 = np.array([4.994095688521491710e-05, 9.667323676989730136e-03, 7.807031992084029071e+01, 4.078837148424790800e+07])





# MSE plotting

fig = plt.figure()
plt.semilogy()

marker = 'X'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof1) / true_Z, label="AHTIS with fixed $\\nu$, $\\nu=1$", marker=marker, linestyle='-', color=color)

marker = '*'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof2) / true_Z, label="AHTIS with fixed $\\nu$, $\\nu=2$", marker=marker, linestyle='-', color=color)

marker = 'o'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof3) / true_Z, label="AHTIS with fixed $\\nu$, $\\nu=3$", marker=marker, linestyle='-', color=color)

marker = 's'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof5) / true_Z, label="AHTIS with fixed $\\nu$, $\\nu=5$", marker=marker, linestyle='-', color=color)

# plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof10) / true_Z, label="AHTIS with fixed $\\nu$, $\\nu=1$0", marker="*")

marker = 'o'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_AMIS_dof3) / true_Z, label="AMIS, $\\nu=3$", linestyle="--", marker=marker, color=color)

marker = 's'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_AMIS_dof5) / true_Z, label="AMIS, $\\nu=5$", linestyle="--", marker="s", color=color)

# plt.plot(d_collect, np.sqrt(MSE_Z_AMIS_dof10) / true_Z, label="AMIS, $\\nu=10$", linestyle="dashed", marker="*")

marker = 'D'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_adaptive) / true_Z, label="AHTIS", marker=marker, linestyle='-', color=color)

plt.xticks(d_collect)

# plt.legend(loc='upper left')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

plt.grid(which='both')  # showing both major and minor grid lines

plt.xlabel(r"Dimension $d$")
plt.ylabel(r"$\sqrt{\mathrm{MSE}} / Z_{\pi}$")

plt.savefig("synthetic_MSE_Z_"+targetName+".pdf",bbox_inches="tight")



# alpha ESS plotting

fig = plt.figure()
plt.semilogy()




marker = 'X'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof1, label="AHTIS with fixed $\\nu$, $\\nu=1$", marker=marker, linestyle="-",color=color)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof1 - alphaESS_std_escortAMIS_dof1, alphaESS_mean_escortAMIS_dof1 + alphaESS_std_escortAMIS_dof1, alpha=0.2, color=color)

marker = '*'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof2, label="AHTIS with fixed $\\nu$, $\\nu=2$", marker=marker, linestyle="-",color=color)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof2 - alphaESS_std_escortAMIS_dof2, alphaESS_mean_escortAMIS_dof2 + alphaESS_std_escortAMIS_dof2, alpha=0.2, color=color)

marker = 'o'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof3, label="AHTIS with fixed $\\nu$, $\\nu=3$", marker=marker, linestyle="-",color=color)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof3 - alphaESS_std_escortAMIS_dof3, alphaESS_mean_escortAMIS_dof3 + alphaESS_std_escortAMIS_dof3, alpha=0.2, color=color)

marker = 's'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof5, label="AHTIS with fixed $\\nu$, $\\nu=5$", marker=marker, linestyle="-",color=color)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof5 - alphaESS_std_escortAMIS_dof5, alphaESS_mean_escortAMIS_dof5 + alphaESS_std_escortAMIS_dof5, alpha=0.2, color=color)

# plt.plot(d_collect, alphaESS_mean_escortAMIS_dof10, label="AHTIS with fixed $\\nu$, $\\nu=1$0", marker="*")

marker = 'o'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_AMIS_dof3, label="AMIS, $\\nu=3$", linestyle="--", marker=marker,color=color)
plt.fill_between(d_collect, alphaESS_mean_AMIS_dof3 - alphaESS_std_AMIS_dof3, alphaESS_mean_AMIS_dof3 + alphaESS_std_AMIS_dof3, alpha=0.2, color=color)

marker = 's'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_AMIS_dof5, label="AMIS, $\\nu=5$", linestyle="--", marker=marker,color=color)
plt.fill_between(d_collect, alphaESS_mean_AMIS_dof5 - alphaESS_std_AMIS_dof5, alphaESS_mean_AMIS_dof5 + alphaESS_std_AMIS_dof5, alpha=0.2, color=color)

# plt.plot(d_collect, alphaESS_mean_AMIS_dof10, label="AMIS, $\\nu=10$", linestyle="dashed", marker="*")


# plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof10 - alphaESS_std_escortAMIS_dof10, alphaESS_mean_escortAMIS_dof10 + alphaESS_std_escortAMIS_dof10, alpha=0.2)

# plt.fill_between(d_collect, alphaESS_mean_AMIS_dof10 - alphaESS_std_AMIS_dof10, alphaESS_mean_AMIS_dof10 + alphaESS_std_AMIS_dof10, alpha=0.2)



marker = 'D'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_adaptive, label="AHTIS", marker=marker, linestyle="-",color=color)
plt.fill_between(d_collect, alphaESS_mean_adaptive - alphaESS_std_adaptive, alphaESS_mean_adaptive + alphaESS_std_adaptive, alpha=0.2)

plt.xticks(d_collect)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)
plt.grid(which='both')  # showing both major and minor grid lines
plt.xlabel("Dimension $d$")
plt.ylabel("$\\alpha$-ESS")

print(fig.get_size_inches())

plt.savefig("synthetic_alphaESS_"+targetName+".pdf",bbox_inches="tight")