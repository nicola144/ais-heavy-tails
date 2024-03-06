import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FormatStrFormatter

plt.rc('text', usetex=True)

plt.rc('font', family='serif', size=22)
plt.rc('axes', labelsize=30, titlesize=20, labelpad=20)  # Adjusting axes parameters
plt.rc('xtick', labelsize=25)  # Adjusting xtick parameters
plt.rc('ytick', labelsize=25)  # Adjusting ytick parameters
plt.rc('legend', handlelength=2, fontsize=25)  # e.g., shorter lines
plt.rc('lines', markersize=8)  # Replace 10 with your desired size

#%%

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

d_collect = [2,4,8,16,32]


# # target dof equal to 2
# true_Z = np.array([1.404962946208144459e+01, 9.869604401089360124e+01, 1.623484850566705973e+03, 3.765290085742292285e+04, 1.101585814280428895e+05])
# targetName = "dofTarget2"
#
# alphaESS_mean_escortAMIS_dof1 = np.array([9.144814464849421176e-01, 8.981825945966060987e-01, 8.880906695810214524e-01, 8.826532506289761493e-01, 8.781449739576143720e-01])
# alphaESS_mean_escortAMIS_dof2 = np.array([9.999913080593623160e-01, 9.999752324964497063e-01, 9.999035903480074650e-01, 9.996008552874264819e-01, 9.981973067759681850e-01])
# alphaESS_mean_escortAMIS_dof3 = np.array([9.703107581569057061e-01, 9.588732348477198020e-01, 9.521499686254339956e-01, 9.473129140156296701e-01, 9.429653924434002743e-01])
# alphaESS_mean_escortAMIS_dof5 = np.array([8.702389160398555923e-01, 8.337632079177429434e-01, 7.820506228753225653e-01, 7.626158036003678475e-01, 7.321564837902565515e-01])
# alphaESS_mean_escortAMIS_dof10 = np.array([7.825653887515278129e-01, 6.958181386442454652e-01, 5.805113123272713782e-01, 5.076722166787422941e-01, 4.327941103859319072e-01])
#
# alphaESS_std_escortAMIS_dof1 = np.array([1.876491026996742612e-03, 1.814864203198255331e-03, 2.034501142463264901e-03, 1.876746922200428101e-03, 1.881578661732201223e-03])
# alphaESS_std_escortAMIS_dof2 = np.array([6.136402240386612833e-06, 9.787650853634096305e-06, 2.021958748877326046e-05, 4.507892638231119129e-05, 1.356152756189015439e-04])
# alphaESS_std_escortAMIS_dof3 = np.array([4.830985918140424246e-03, 7.246578048611529779e-03, 4.862451845208858904e-03, 4.625628624005758208e-03, 5.949974764362010553e-03])
# alphaESS_std_escortAMIS_dof5 = np.array([1.041486667041957304e-01, 4.374593000605125209e-02, 4.589682622705657661e-02, 4.587162275060568950e-02, 4.991253711633970164e-02])
# alphaESS_std_escortAMIS_dof10 = np.array([1.400719423505334404e-01, 1.007743183913678009e-01, 1.141211380411730758e-01, 1.053985939329236815e-01, 1.202333752575649034e-01])
#
# MSE_Z_escortAMIS_dof1 = np.array([1.155109970719065895e-04, 8.328016828798572024e-03, 2.827138705627340265e+00, 3.060072055903096953e+03, 3.202404891878904309e+05])
# MSE_Z_escortAMIS_dof2 = np.array([1.283719866724547356e-05, 5.155852660450637370e-04, 3.728151849836758847e-01, 1.263489762824257241e+03, 2.095956350928405300e+05])
# MSE_Z_escortAMIS_dof3 = np.array([5.810489344318569757e-05, 3.759686412102727741e-03, 1.959848786973888091e+00, 2.301547826966623688e+03, 3.304084140680228011e+05])
# MSE_Z_escortAMIS_dof5 = np.array([8.638888066407172679e-04, 6.519081092903490637e-02, 3.192798568028252859e+01, 4.802412997816570714e+04, 3.855124986351326574e+06])
# MSE_Z_escortAMIS_dof10 = np.array([7.490752351310303592e-03, 9.753313423075428146e-01, 7.459457109121632357e+02, 1.477757177243437385e+06, 8.209243482378529012e+07])
#
#
# alphaESS_mean_adaptive = np.array([9.923895884629682085e-01, 9.870077633517324589e-01, 9.920454581998857657e-01, 9.910579921332822728e-01, 9.958613502919161364e-01])
# alphaESS_std_adaptive = np.array([2.344605314529271331e-02, 4.910411401587817110e-02, 3.810206097003925041e-02, 4.898319924375282852e-02, 4.517840781143636283e-03])
# MSE_Z_adaptive = np.array([1.366206052759713757e-05, 9.263725499640704459e-04, 5.740420487947989203e-01, 1.558374659628888821e+03, 2.066400478909699596e+05])
#
#
# alphaESS_mean_AMIS_dof3 = np.array([7.659374575652233563e-01, 6.664563544084158853e-01, 5.786582472080282757e-01, 5.797713148235533742e-01, 6.586390860647435286e-01])
# alphaESS_mean_AMIS_dof5 = np.array([7.015030943837191613e-01, 5.256082349212668481e-01, 3.818471395949649683e-01, 3.048852071993036206e-01, 3.185774102868152435e-01])
# alphaESS_mean_AMIS_dof10 = np.array([7.009875561134312205e-01, 5.478054410991100021e-01, 3.519177320376049023e-01, 2.522118632814392414e-01, 2.742252740654528287e-01])
#
# alphaESS_std_AMIS_dof3 = np.array([6.558522800371312245e-02, 5.021573768778590102e-02, 7.060628607807720170e-02, 7.351717874973173750e-02, 7.880304120620069963e-02])
# alphaESS_std_AMIS_dof5 = np.array([4.185772994488935056e-02, 5.710959651686075417e-02, 4.489901576815884254e-02, 8.204959792239004124e-02, 1.207547523894243191e-01])
# alphaESS_std_AMIS_dof10 = np.array([7.838414445943911457e-02, 6.444046138159498227e-02, 7.817927393964584082e-02, 8.422124797246401118e-02, 1.014464656118576474e-01])
#
# MSE_Z_AMIS_dof3 = np.array([4.468962243240218266e-04, 5.629800319176823248e-02, 2.215160901061101484e+01, 3.007291938423297688e+04, 7.770635965873304522e+05])
# MSE_Z_AMIS_dof5 = np.array([6.829300353220114786e-04, 1.428253062657661721e-01, 1.354714861697121364e+02, 5.795708411302637542e+05, 1.586847642032048665e+07])
# MSE_Z_AMIS_dof10 = np.array([5.241437234267612086e-03, 8.462517690205006993e-01, 6.275423370822865081e+02, 2.279684348320381250e+06, 9.806025992916481197e+07])
#

# target dof equal to 5
true_Z = np.array([1.409943485869908386e+02, 1.409943485869908386e+02, 7.028072946176218466e+03, 2.717306491258586291e+06, 4.760198307706989288e+09])
targetName = "dofTarget5"

alphaESS_mean_escortAMIS_dof1 = np.array([7.934226127952928032e-01, 7.310923811449652199e-01, 6.816655391996948588e-01, 6.489872146801646258e-01, 6.265572150914213712e-01])
alphaESS_mean_escortAMIS_dof2 = np.array([9.392924600718375316e-01, 9.098510660575870324e-01, 8.815974452894139723e-01, 8.600488035754887406e-01, 8.433072923519999708e-01])
alphaESS_mean_escortAMIS_dof3 = np.array([9.833134463029642447e-01, 9.730357572222129159e-01, 9.617637198980206747e-01, 9.520765087739251342e-01, 9.436052863188640893e-01])
alphaESS_mean_escortAMIS_dof5 = np.array([9.999903890008293716e-01, 9.999699781629713957e-01, 9.998944452362458213e-01, 9.995926554995425706e-01, 9.975508646038498961e-01])
alphaESS_mean_escortAMIS_dof10 = np.array([9.821249630500021865e-01, 9.668297702504521363e-01, 9.368140838818128646e-01, 9.074190736738451113e-01, 8.803938395333461564e-01])

alphaESS_std_escortAMIS_dof1 = np.array([2.806880437443453735e-03, 3.347018290721677850e-03, 3.511392104322683637e-03, 3.570272544710990058e-03, 3.468334721350044367e-03])
alphaESS_std_escortAMIS_dof2 = np.array([1.599987824125829409e-03, 1.903984435380082088e-03, 2.055018412790969411e-03, 2.335316776825327630e-03, 2.406112500565875823e-03])
alphaESS_std_escortAMIS_dof3 = np.array([5.763685637842146884e-04, 9.240119711780532064e-04, 1.013819946176315037e-03, 9.813289938449770797e-04, 1.433406692853281391e-03])
alphaESS_std_escortAMIS_dof5 = np.array([6.513296917554514887e-06, 1.093678566449271963e-05, 1.940672622523351067e-05, 4.414081810184986249e-05, 3.046257974451181227e-04])
alphaESS_std_escortAMIS_dof10 = np.array([6.862363906509987531e-03, 1.025794507587121304e-02, 4.320897541740194997e-02, 3.025245245002826378e-02, 2.568782896488885914e-02])

MSE_Z_escortAMIS_dof1 = np.array([5.331004458700579988e-04, 4.813709540803217529e-02, 1.104990301540095743e+02, 4.597810320877866447e+07, 3.177590590504873500e+15])
MSE_Z_escortAMIS_dof2 = np.array([7.869276789534255276e-05, 1.544162448249263020e-02, 4.249017775722693102e+01, 1.965758856025061384e+07, 1.404368752620385500e+15])
MSE_Z_escortAMIS_dof3 = np.array([3.999427501772853091e-05, 6.293553464937198458e-03, 2.058921455179503113e+01, 1.009652632730942592e+07, 9.405799643573310000e+14])
MSE_Z_escortAMIS_dof5 = np.array([1.312796543613250550e-05, 1.378591416159999014e-03, 7.934710989683143367e+00, 6.285406252703314647e+06, 8.410556653565755000e+14])
MSE_Z_escortAMIS_dof10 = np.array([3.893314055018779558e-05, 7.287334515395931141e-03, 4.943531595714093640e+01, 2.765861453500132263e+07, 2.985292519078642500e+15])


alphaESS_mean_adaptive = np.array([9.999006891850575895e-01, 9.998496141254210601e-01, 9.993099756738911266e-01, 9.915802285376412728e-01, 9.955259205570095027e-01])
alphaESS_std_adaptive = np.array([2.186818811641501662e-04, 1.896168337427528786e-04, 1.155923309316938055e-03, 3.593198467896641796e-02, 2.982201448216848481e-03])
MSE_Z_adaptive = np.array([2.019963716080580004e-05, 1.312256467378856246e-03, 7.806592116210620880e+00, 7.184603819826675579e+06, 7.193185301910256250e+14])


alphaESS_mean_AMIS_dof3 = np.array([9.267350999215879792e-01, 8.839430186217240859e-01, 8.406173903007153614e-01, 8.053579847616778586e-01, 7.761278946296522596e-01])
alphaESS_mean_AMIS_dof5 = np.array([9.999739357129183714e-01, 9.999254711304466081e-01, 9.997479123107883003e-01, 9.990422163132122169e-01, 9.944579186057223019e-01])
alphaESS_mean_AMIS_dof10 = np.array([9.780486435922012278e-01, 9.544831835554397070e-01, 9.098606060494455194e-01, 8.569377252709959869e-01, 8.130581127962104304e-01])

alphaESS_std_AMIS_dof3 = np.array([8.828580536520690440e-04, 1.491191707571309488e-03, 2.172356185066151926e-03, 2.695195898399960818e-03, 3.104259732209939220e-03])
alphaESS_std_AMIS_dof5 = np.array([1.859939569304494241e-05, 3.442251978639737864e-05, 6.683963635418733648e-05, 1.369821477404313634e-04, 8.344886701799874434e-04])
alphaESS_std_AMIS_dof10 = np.array([5.461498387203026113e-03, 6.514856450486256230e-03, 1.369703667479143551e-02, 1.309882876782711078e-02, 5.958598530878022465e-02])

MSE_Z_AMIS_dof3 = np.array([1.169571818936132612e-04, 1.874001588714309732e-02, 8.796354567525187917e+01, 2.322256702357692271e+07, 1.399067202212805500e+15])
MSE_Z_AMIS_dof5 = np.array([9.360621769185713969e-06, 1.387316748219458162e-03, 4.683516993399246431e+00, 5.815979157076846808e+06, 7.416215945331710000e+14])
MSE_Z_AMIS_dof10 = np.array([4.994095688521491710e-05, 9.667323676989730136e-03, 7.807031992084029071e+01, 4.078837148424790800e+07, 3.388353343840937000e+15])




# MSE plotting

fig = plt.figure()
fig.set_size_inches((6.4,4.8))
plt.semilogy()

# marker = 'X'
# color = get_color_for_marker(marker, marker_to_color, color_cycle)
# plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof1) / true_Z, label=r"AHTIS with fixed $\nu$, $\nu=1$", marker=marker, linestyle='-', color=color)
#
# marker = '*'
# color = get_color_for_marker(marker, marker_to_color, color_cycle)
# plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof2) / true_Z, label=r"AHTIS with fixed $\nu$, $\nu=2$", marker=marker, linestyle='-', color=color)
#


marker = 'o'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof3) / true_Z, label=r"AHTIS with fixed $\nu$, $\nu=3$", marker=marker, linestyle='-', color=color)

marker = 's'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof5) / true_Z, label=r"AHTIS with fixed $\nu$, $\nu=5$", marker=marker, linestyle='-', color=color)

marker = 'X'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_escortAMIS_dof10) / true_Z, label=r"AHTIS with fixed $\nu$, $\nu=10$", marker=marker, linestyle='-', color=color)

marker = 'o'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_AMIS_dof3) / true_Z, label=r"AMIS, $\nu=3$", linestyle="--", marker=marker, color=color)

marker = 's'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_AMIS_dof5) / true_Z, label=r"AMIS, $\nu=5$", linestyle="--", marker="s", color=color)

marker = 'X'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_AMIS_dof10) / true_Z, label=r"AMIS, $\nu=10$", marker=marker, linestyle='-', color=color)

marker = 'D'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, np.sqrt(MSE_Z_adaptive) / true_Z, label=r"AHTIS", marker=marker, linestyle='-', color=color)

plt.xticks(d_collect)

# plt.legend(loc='upper left')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=2)

plt.grid(which='both')  # showing both major and minor grid lines

plt.xlabel(r"Dimension $d$")
plt.ylabel(r"$\sqrt{\mathrm{MSE}} / Z_{\pi}$")
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

plt.savefig("synthetic_MSE_Z_"+targetName+".pdf",bbox_inches="tight")



# alpha ESS plotting

fig = plt.figure()
# plt.semilogy()
fig.set_size_inches(( 6.4, 4.8))


# Disable scientific notation for x-axis
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_xaxis().get_major_formatter().set_scientific(False)


# marker = 'X'
# color = get_color_for_marker(marker, marker_to_color, color_cycle)
# plt.plot(d_collect, alphaESS_mean_escortAMIS_dof1, label=r"AHTIS with fixed $\nu$, $\nu=1$", marker=marker, linestyle="-",color=color)
# plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof1 - alphaESS_std_escortAMIS_dof1, alphaESS_mean_escortAMIS_dof1 + alphaESS_std_escortAMIS_dof1, alpha=0.2, color=color)
#
# marker = '*'
# color = get_color_for_marker(marker, marker_to_color, color_cycle)
# plt.plot(d_collect, alphaESS_mean_escortAMIS_dof2, label=r"AHTIS with fixed $\nu$, $\nu=2$", marker=marker, linestyle="-",color=color)
# plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof2 - alphaESS_std_escortAMIS_dof2, alphaESS_mean_escortAMIS_dof2 + alphaESS_std_escortAMIS_dof2, alpha=0.2, color=color)

marker = 'o'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof3, label=r"AHTIS with fixed $\nu$, $\nu=3$", marker=marker, linestyle="-",color=color)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof3 - alphaESS_std_escortAMIS_dof3, alphaESS_mean_escortAMIS_dof3 + alphaESS_std_escortAMIS_dof3, alpha=0.2, color=color)

marker = 's'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof5, label=r"AHTIS with fixed $\nu$, $\nu=5$", marker=marker, linestyle="-",color=color)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof5 - alphaESS_std_escortAMIS_dof5, alphaESS_mean_escortAMIS_dof5 + alphaESS_std_escortAMIS_dof5, alpha=0.2, color=color)

marker = 'X'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_escortAMIS_dof10, label=r"AHTIS with fixed $\nu$, $\nu=10$", marker=marker, linestyle="-",color=color)
plt.fill_between(d_collect, alphaESS_mean_escortAMIS_dof10 - alphaESS_std_escortAMIS_dof10, alphaESS_mean_escortAMIS_dof10 + alphaESS_std_escortAMIS_dof10, alpha=0.2, color=color)

marker = 'o'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_AMIS_dof3, label=r"AMIS, $\nu=3$", linestyle="--", marker=marker,color=color)
plt.fill_between(d_collect, alphaESS_mean_AMIS_dof3 - alphaESS_std_AMIS_dof3, alphaESS_mean_AMIS_dof3 + alphaESS_std_AMIS_dof3, alpha=0.2, color=color)

marker = 's'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_AMIS_dof5, label=r"AMIS, $\nu=5$", linestyle="--", marker=marker,color=color)
plt.fill_between(d_collect, alphaESS_mean_AMIS_dof5 - alphaESS_std_AMIS_dof5, alphaESS_mean_AMIS_dof5 + alphaESS_std_AMIS_dof5, alpha=0.2, color=color)

marker = 'X'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_AMIS_dof10, label=r"AMIS, $\nu=10$", linestyle="--", marker=marker,color=color)
plt.fill_between(d_collect, alphaESS_mean_AMIS_dof10 - alphaESS_std_AMIS_dof10, alphaESS_mean_AMIS_dof10 + alphaESS_std_AMIS_dof10, alpha=0.2, color=color)



marker = 'D'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(d_collect, alphaESS_mean_adaptive, label=r"AHTIS", marker=marker, linestyle="-",color=color)
plt.fill_between(d_collect, alphaESS_mean_adaptive - alphaESS_std_adaptive, alphaESS_mean_adaptive + alphaESS_std_adaptive, alpha=0.2, color=color)


plt.xticks(d_collect)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=2)
plt.grid(which='both')  # showing both major and minor grid lines
plt.xlabel(r"Dimension $d$")
plt.ylabel(r"$\alpha$-ESS")
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

print(fig.get_size_inches())

plt.savefig("synthetic_alphaESS_"+targetName+".pdf",bbox_inches="tight")

################################################################################
#### New






nb_runs = 250 # was 150 for good res,



# load
def load_all_npy_files(directory):
    """
    Load all .npy files from a specified directory.

    Parameters:
    - directory (str): Path to the directory containing .npy files.

    Returns:
    - A dictionary with filenames as keys and loaded numpy arrays as values.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    data = {}

    for f in files:
        full_path = os.path.join(directory, f)
        data[f] = np.load(full_path)

    return data

# Usage
directory_path = './results/realdata/all_replicates/AMIS'
loaded_data = load_all_npy_files(directory_path)


directory_path_alphaAMIS = './results/realdata/all_replicates/alphaAMIS/'
loaded_data_alphaAMIS = load_all_npy_files(directory_path_alphaAMIS)

directory_path_adaptivealphaAMIS = './results/realdata/all_replicates/adaptiveGPadaptivealphaAMIS/'
loaded_data_adaptivealphaAMIS = load_all_npy_files(directory_path_adaptivealphaAMIS)

dictionaries_data = [loaded_data, loaded_data_alphaAMIS, loaded_data_adaptivealphaAMIS]

set_keys_metrics, set_dofs, set_initialisation, set_nbruns, set_numsamples = set(), set(), set(), set(), set()

# First create dict
for i, data in enumerate(dictionaries_data):
    for key, _ in data.items():

        parts = key.replace(".npy", "").split("_")
        cleaned_parts = list(filter(lambda item: item != '', parts))

        # print(cleaned_parts)

        initialisation = cleaned_parts[0]
        numsamples = cleaned_parts[1]
        metric = cleaned_parts[-1] if (i==1) or (i ==2) else cleaned_parts[-2]

        # if metric == 'alldof':
        #     continue

        dof = cleaned_parts[3]
        nbruns = cleaned_parts[5]

        set_keys_metrics.add(metric)
        set_dofs.add(dof)
        set_initialisation.add(initialisation)
        set_numsamples.add(numsamples)
        set_nbruns.add(nbruns)

results_final_baseline, results_final_ours, results_final_ours_adapted = {}, {}, {}

set_keys_metrics_no_dofs = set_keys_metrics.copy()
set_keys_metrics_no_dofs.remove('alldof')

for metric in set_keys_metrics_no_dofs:
    results_final_baseline[metric], results_final_ours[metric] = {}, {}
    for initt in set_initialisation:
        results_final_baseline[metric][initt], results_final_ours[metric][initt] = {}, {}
        for dof in set_dofs:
            results_final_baseline[metric][initt][dof], results_final_ours[metric][initt][dof] = {}, {}
            for nbruns in set_nbruns:
                results_final_baseline[metric][initt][dof][nbruns], results_final_ours[metric][initt][dof][nbruns] = {}, {}
                for numsamples in set_numsamples:
                    results_final_baseline[metric][initt][dof][nbruns][numsamples], results_final_ours[metric][initt][dof][nbruns][numsamples]= None, None

for metric in set_keys_metrics:
    results_final_ours_adapted[metric] = {}
    for initt in set_initialisation:
        results_final_ours_adapted[metric][initt] = {}
        for dof in set_dofs:
            results_final_ours_adapted[metric][initt][dof] = {}
            for nbruns in set_nbruns:
                results_final_ours_adapted[metric][initt][dof][nbruns] = {}
                for numsamples in set_numsamples:
                    results_final_ours_adapted[metric][initt][dof][nbruns][numsamples] = None

for i, data in enumerate(dictionaries_data):
    for key, value in data.items():

        parts = key.replace(".npy", "").split("_")
        cleaned_parts = list(filter(lambda item: item != '', parts))

        initialisation = cleaned_parts[0]
        numsamples = cleaned_parts[1]
        metric = cleaned_parts[-1] if (i==1) or (i ==2) else cleaned_parts[-2]

        dof = cleaned_parts[3]
        nbruns = cleaned_parts[5]

        if i == 0:
            results_final_baseline[metric][initialisation][dof][nbruns][numsamples] = value
        elif i == 1:
            results_final_ours[metric][initialisation][dof][nbruns][numsamples] = value
        else: # i == 2
            results_final_ours_adapted[metric][initialisation][dof][nbruns][numsamples] = value


# # Cleaning for old data
# # First create dict
# for i, data in enumerate(dictionaries_data):
#     for key, _ in data.items():
#
#         parts = key.replace(".npy", "").replace("niter","").replace("25","").split("_")
#         cleaned_parts = list(filter(lambda item: item != '', parts))
#
#         # Cleaning
#         if 'ESS' in cleaned_parts:
#             idx = cleaned_parts.index('ESS')
#             cleaned_parts[idx-1]+= 'ESS'
#             cleaned_parts.remove('ESS')
#         if 'alphaESS' in cleaned_parts:
#             idx = cleaned_parts.index('alphaESS')
#             cleaned_parts[idx-1]+= 'alphaESS'
#             cleaned_parts.remove('alphaESS')
#         if 'Z' in cleaned_parts:
#             idx = cleaned_parts.index('Z')
#             cleaned_parts[idx-1]+= 'Z'
#             cleaned_parts.remove('Z')
#
#         initialisation = cleaned_parts[0]
#         numsamples = cleaned_parts[1]
#         metric = cleaned_parts[-1] if i==1 else cleaned_parts[-2]
#         dof = cleaned_parts[3]
#         nbruns = cleaned_parts[5]
#
#         set_keys_metrics.add(metric)
#         set_dofs.add(dof)
#         set_initialisation.add(initialisation)
#         set_numsamples.add(numsamples)
#         set_nbruns.add(nbruns)
#
# results_final_baseline, results_final_ours= {}, {}
#
# for metric in set_keys_metrics:
#     results_final_baseline[metric], results_final_ours[metric] = {}, {}
#     for initt in set_initialisation:
#         results_final_baseline[metric][initt], results_final_ours[metric][initt] = {}, {}
#         for dof in set_dofs:
#             results_final_baseline[metric][initt][dof], results_final_ours[metric][initt][dof] = {}, {}
#             for nbruns in set_nbruns:
#                 results_final_baseline[metric][initt][dof][nbruns], results_final_ours[metric][initt][dof][nbruns] = {}, {}
#                 for numsamples in set_numsamples:
#                     results_final_baseline[metric][initt][dof][nbruns][numsamples], results_final_ours[metric][initt][dof][nbruns][numsamples] = [], []
#
# for i, data in enumerate(dictionaries_data):
#     for key, value in data.items():
#
#         parts = key.replace(".npy", "").replace("niter","").replace("25","").split("_")
#         cleaned_parts = list(filter(lambda item: item != '', parts))
#
#         # Cleaning
#         if 'ESS' in cleaned_parts:
#             idx = cleaned_parts.index('ESS')
#             cleaned_parts[idx-1]+= 'ESS'
#             cleaned_parts.remove('ESS')
#         if 'alphaESS' in cleaned_parts:
#             idx = cleaned_parts.index('alphaESS')
#             cleaned_parts[idx-1]+= 'alphaESS'
#             cleaned_parts.remove('alphaESS')
#         if 'Z' in cleaned_parts:
#             idx = cleaned_parts.index('Z')
#             cleaned_parts[idx-1]+= 'Z'
#             cleaned_parts.remove('Z')
#
#         initialisation = cleaned_parts[0]
#         numsamples = cleaned_parts[1]
#         metric = cleaned_parts[-1] if i==1 else cleaned_parts[-2]
#         dof = cleaned_parts[3]
#         nbruns = cleaned_parts[5]
#
#         if not value.shape == ():
#             value = value[-1] # keeping only last value of iteration number
#
#         if i == 0:
#             # print(value)
#             # print(value.item())
#             # print(value.shape)
#             results_final_baseline[metric][initialisation][dof][nbruns][numsamples].append(value.item())
#         else: #(i == 1)
#             results_final_ours[metric][initialisation][dof][nbruns][numsamples].append(value.item())



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





from matplotlib.ticker import MultipleLocator

fig = plt.figure()
fig.set_size_inches(( 6.4, 4.8))

# plt.rc('font', family='serif', size=18)
# plt.rc('axes', labelsize=25, titlesize=20, labelpad=20)  # Adjusting axes parameters
# plt.rc('xtick', labelsize=20)  # Adjusting xtick parameters
# plt.rc('ytick', labelsize=20)  # Adjusting ytick parameters
# plt.rc('legend', fontsize=12, loc='lower left')  # Adjusting legend parameters
# plt.rc('legend', handlelength=1)  # e.g., shorter lines


for dof in ['3','4','5']:

    sorted_numsamples_baseline = sorted(
        [
            int(key)
            for key, value in results_final_baseline['alphaESS']['Sigmfixedinit4'][dof][str(nb_runs)].items()
            if value is not None
        ]
    )
    sorted_numsamples_baseline_str = [str(num) for num in sorted_numsamples_baseline]

    # Step 2: Iterate through the sorted keys and get the corresponding values
    # all_replicates_baseline = np.asarray([ results_final_baseline['alphaESS']['Sigmfixedinit4'][dof][str(nb_runs)][key].mean() for key in sorted_numsamples_baseline_str])
    sorted_values_baseline = []
    for key in sorted_numsamples_baseline_str:
            mean_value = results_final_baseline['alphaESS']['Sigmfixedinit4'][dof][str(nb_runs)][key][:,-1].mean()
            sorted_values_baseline.append(mean_value)

    sorted_values_baseline = np.asarray(sorted_values_baseline)

    sorted_values_baseline_std = []
    for key in sorted_numsamples_baseline_str:
            std_val = results_final_baseline['alphaESS']['Sigmfixedinit4'][dof][str(nb_runs)][key][:,-1].std()
            sorted_values_baseline_std.append(std_val)

    sorted_values_baseline_std = np.asarray(sorted_values_baseline_std)



    sorted_numsamples_ours = sorted(
        [
            int(key)
            for key, value in results_final_ours['alphaESS']['Sigmfixedinit4'][dof][str(nb_runs)].items()
            if value is not None
        ]
    )
    sorted_numsamples_ours_str = [str(num) for num in sorted_numsamples_ours]

    # Step 2: Iterate through the sorted keys and get the corresponding values

    sorted_values_ours = []
    for key in sorted_numsamples_ours_str:
            mean_value = results_final_ours['alphaESS']['Sigmfixedinit4'][dof][str(nb_runs)][key][:,-1].mean()
            sorted_values_ours.append(mean_value)

    sorted_values_ours = np.asarray(sorted_values_ours)

    sorted_values_ours_std = []
    for key in sorted_numsamples_ours_str:
            std_val = results_final_ours['alphaESS']['Sigmfixedinit4'][dof][str(nb_runs)][key][:,-1].std()
            sorted_values_ours_std.append(std_val)

    sorted_values_ours_std = np.asarray(sorted_values_ours_std)

    if dof == '3':
        marker = 'o'
    elif dof == '4':
        marker = 'v'
    else:
        marker = 's'

    color = get_color_for_marker(marker, marker_to_color, color_cycle)
    plt.plot(sorted_numsamples_baseline,  sorted_values_baseline,  marker=marker, markevery=5, linestyle='--', label=r'AMIS, $\nu={}$'.format(dof), color=color)
    plt.plot(sorted_numsamples_ours,  sorted_values_ours, marker=marker, markevery=5, linestyle='-', label=r'AHTIS with fixed $\nu$, $\nu={}$'.format(dof), color=color)

    plt.fill_between(sorted_numsamples_baseline, sorted_values_baseline - sorted_values_baseline_std, sorted_values_baseline + sorted_values_baseline_std, alpha=0.1, color=color)
    plt.fill_between(sorted_numsamples_ours, sorted_values_ours - sorted_values_ours_std, sorted_values_ours + sorted_values_ours_std, alpha=0.1, color=color)


# Adapted dof (recall it is stored in a fixed dof = 3 for now)
sorted_numsamples_ours_adapted = sorted(
    [
        int(key)
        for key, value in results_final_ours_adapted['alphaESS']['Sigmfixedinit4']['3'][str(nb_runs)].items()
        if value is not None
    ]
)

sorted_numsamples_ours_adapted_str = [str(num) for num in sorted_numsamples_ours_adapted]

# Step 2: Iterate through the sorted keys and get the corresponding values

sorted_values_ours_adapted = []
for key in sorted_numsamples_ours_adapted_str:
        mean_value = results_final_ours_adapted['alphaESS']['Sigmfixedinit4']['3'][str(nb_runs)][key][:,-1].mean()
        sorted_values_ours_adapted.append(mean_value)

sorted_values_ours_adapted = np.asarray(sorted_values_ours_adapted)

sorted_values_ours_adapted_std = []
for key in sorted_numsamples_ours_adapted_str:
        std_val = results_final_ours_adapted['alphaESS']['Sigmfixedinit4']['3'][str(nb_runs)][key][:,-1].std()
        sorted_values_ours_adapted_std.append(std_val)

sorted_values_ours_adapted_std = np.asarray(sorted_values_ours_adapted_std)

marker = 'D'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(sorted_numsamples_ours_adapted,  sorted_values_ours_adapted,  marker=marker, markevery=5, linestyle='-', label='AHTIS ', color=color)

plt.fill_between(sorted_numsamples_ours_adapted, sorted_values_ours_adapted - sorted_values_ours_adapted_std, sorted_values_ours_adapted +  sorted_values_ours_adapted_std, alpha=0.1, color=color)

# Access the current Axes instance:
ax = plt.gca()

# Using MultipleLocator to place a tick every 0.5 units
ax.xaxis.set_major_locator(MultipleLocator(500))

plt.xticks(sorted_numsamples_ours_adapted)
# plt.xticks(rotation=90)

# plt.yscale('log')  # Set y-axis to log scale
# plt.xscale('log')  # Set y-axis to log scale

# plt.grid(axis='y')  # showing both major and minor grid lines

# Get current x-tick labels
xticks_locs, xticks_old = plt.xticks()

# Create new x-tick labels
new_xticks = [''] * len(xticks_locs)
for i, loc in enumerate(xticks_locs):
    # Show every 5th, the first, and the last tick label
    if i % 5 == 0 or i == 0 or i == len(xticks_locs)-1:
        new_xticks[i] = str(int(loc))
plt.xticks(xticks_locs, new_xticks)


plt.xlabel('Number of samples', fontsize=35)
plt.ylabel(r'$\alpha$-ESS', fontsize=35)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# plt.axhline(y=truth, color='r', linestyle='solid', lw=2, label=r'AMIS with $10^5$ samples (dof=5)')
plt.tight_layout()
plt.grid(which='both')  # showing both major and minor grid lines
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.47), ncol=2)
plt.savefig('./results/creatinine_alphaESS.pdf',bbox_inches='tight')



# load ground truth
directory_path_old = './results/realdata/AMIS'
loaded_dataOld = load_all_npy_files(directory_path_old)

set_keys_metricsOld, set_dofsOld, set_initialisationOld, set_nbrunsOld, set_numsamplesOld = set(), set(), set(), set(), set()

for key, _ in loaded_dataOld.items():

    parts = key.replace(".npy", "").replace("niter","").replace("25","").split("_")
    cleaned_parts = list(filter(lambda item: item != '', parts))

    # Cleaning
    if 'ESS' in cleaned_parts:
        idx = cleaned_parts.index('ESS')
        cleaned_parts[idx-1]+= 'ESS'
        cleaned_parts.remove('ESS')
    if 'alphaESS' in cleaned_parts:
        idx = cleaned_parts.index('alphaESS')
        cleaned_parts[idx-1]+= 'alphaESS'
        cleaned_parts.remove('alphaESS')
    if 'Z' in cleaned_parts:
        idx = cleaned_parts.index('Z')
        cleaned_parts[idx-1]+= 'Z'
        cleaned_parts.remove('Z')

    initialisation = cleaned_parts[0]
    numsamples = cleaned_parts[1]
    metric = cleaned_parts[-2]
    dof = cleaned_parts[3]
    nbruns = cleaned_parts[5]


    set_keys_metricsOld.add(metric)
    set_dofsOld.add(dof)
    set_initialisationOld.add(initialisation)
    set_numsamplesOld.add(numsamples)
    set_nbrunsOld.add(nbruns)


results_old_baseline = {}
for metric in set_keys_metricsOld:
    results_old_baseline[metric] = {}
    for initt in set_initialisationOld:
        results_old_baseline[metric][initt] = {}
        for dof in set_dofsOld:
            results_old_baseline[metric][initt][dof] = {}
            for nbruns in set_nbrunsOld:
                results_old_baseline[metric][initt][dof][nbruns] = {}
                for numsamples in set_numsamplesOld:
                    results_old_baseline[metric][initt][dof][nbruns][numsamples] = None

for key, value in loaded_dataOld.items():

    parts = key.replace(".npy", "").replace("niter","").replace("25","").split("_")
    cleaned_parts = list(filter(lambda item: item != '', parts))

    # Cleaning
    if 'ESS' in cleaned_parts:
        idx = cleaned_parts.index('ESS')
        cleaned_parts[idx-1]+= 'ESS'
        cleaned_parts.remove('ESS')
    if 'alphaESS' in cleaned_parts:
        idx = cleaned_parts.index('alphaESS')
        cleaned_parts[idx-1]+= 'alphaESS'
        cleaned_parts.remove('alphaESS')
    if 'Z' in cleaned_parts:
        idx = cleaned_parts.index('Z')
        cleaned_parts[idx-1]+= 'Z'
        cleaned_parts.remove('Z')

    initialisation = cleaned_parts[0]
    numsamples = cleaned_parts[1]
    metric = cleaned_parts[-2]

    dof = cleaned_parts[3]
    nbruns = cleaned_parts[5]

    results_old_baseline[metric][initialisation][dof][nbruns][numsamples] = value



fig = plt.figure()
fig.set_size_inches(( 6.4, 4.8))

# Best DOF it seems
truth_key = '100000'
truth_value = results_old_baseline['meanZ']['Laplaceinit']['5']['1000'][truth_key]

for dof in ['3','4','5']:

    sorted_numsamples_baseline = sorted(
        [
            int(key)
            for key, value in results_final_baseline['allestZ']['Sigmfixedinit4'][dof][str(nb_runs)].items()
            if value is not None
        ]
    )
    sorted_numsamples_baseline_str = [str(num) for num in sorted_numsamples_baseline]

    sorted_numsamples_ours = sorted(
        [
            int(key)
            for key, value in results_final_ours['allestZ']['Sigmfixedinit4'][dof][str(nb_runs)].items()
            if value is not None
        ]
    )

    sorted_numsamples_str = [str(num) for num in sorted_numsamples_ours] # should be the same as baseline right ?



    # Step 2: Iterate through the sorted keys and get the corresponding values
    sorted_values_baseline = [results_final_baseline['allestZ']['Sigmfixedinit4'][dof][str(nb_runs)][key][:,-1] for key in sorted_numsamples_baseline_str]
    sorted_values_ours = [results_final_ours['allestZ']['Sigmfixedinit4'][dof][str(nb_runs)][key][:,-1] for key in sorted_numsamples_str]

    # Calculating the MSE and square root of the relative MSE for baseline and our algorithm
    mse_baseline = [(np.mean((vals - truth_value) ** 2)) for vals in sorted_values_baseline]
    mse_ours = [(np.mean((vals - truth_value) ** 2)) for vals in sorted_values_ours]

    sqrt_rel_mse_baseline = np.sqrt(np.array(mse_baseline) / truth_value)
    sqrt_rel_mse_ours = np.sqrt(np.array(mse_ours) / truth_value)


    if dof == '3':
        marker = 'o'
    elif dof == '4':
        marker = 'v'
    else:
        marker = 's'

    color = get_color_for_marker(marker, marker_to_color, color_cycle)
    # Plotting
    plt.plot(sorted_numsamples_baseline, sqrt_rel_mse_baseline, label=r'AMIS, $\nu$={}'.format(dof), marker=marker, markevery=5, linestyle='--', color=color)
    plt.plot(sorted_numsamples_ours, sqrt_rel_mse_ours, label=r'AHTIS with fixed $\nu$, $\nu$={}'.format(dof), marker=marker, markevery=5, color=color)


sorted_values_ours_adapted = [results_final_ours_adapted['allestZ']['Sigmfixedinit4']['3'][str(nb_runs)][key][:,-1] for key in sorted_numsamples_baseline_str] # should be the same nsamples anyway

mse_ours_adapted = [(np.mean((vals - truth_value) ** 2)) for vals in sorted_values_ours_adapted]

sqrt_rel_mse_ours_adapted = np.sqrt(np.array(mse_ours_adapted) / truth_value)

marker = 'D'
color = get_color_for_marker(marker, marker_to_color, color_cycle)
plt.plot(sorted_numsamples_baseline, sqrt_rel_mse_ours_adapted, label='AHTIS', marker=marker, markevery=5, color=color)


# # Get current y-limits
# ymin, ymax = plt.ylim()
#
# # Get the current axis
# ax = plt.gca()
#
# # Define a function to format the tick labels in scientific notation
# def format_func(value, tick_number):
#     return f"{value:.1e}"
#
# # Apply the function as a formatter for the major and minor y-ticks
# ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_func))
# ax.yaxis.set_minor_formatter(mticker.FuncFormatter(format_func))
#
# # Making sure minor ticks are displayed
# ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
#
# # Set the tick parameters for minor ticks
# ax.tick_params(axis='y', which='minor', size=5)
# ax.tick_params(axis='y', which='major', size=10)
#
# # Adjust the size of tick labels for better readability
# ax.tick_params(axis='y', which='both', labelsize=8)
#
# # Optional: Rotate labels for even better readability
# ax.tick_params(axis='y', which='both', labelrotation=90)
#
# # To ensure that all labels (including minor ticks) are drawn
# ax.minorticks_on()
#
# # Force drawing of minor labels
# ax.yaxis.set_tick_params(which='minor', labelleft=True)

# Get the current axis



# # Get current x-tick labels
# xticks_locs, xticks_old = plt.xticks()
#
# # Create new x-tick labels
# new_xticks = [''] * len(xticks_locs)
# for i, loc in enumerate(xticks_locs):
#     # Show every 5th, the first, and the last tick label
#     if i % 5 == 0 or i == 0 or i == len(xticks_locs)-1:
#         new_xticks[i] = str(int(loc))
# plt.xticks(xticks_locs, new_xticks)

plt.xlabel('Number of samples', fontsize=35)
plt.ylabel(r"$\sqrt{\mathrm{MSE}} / Z_{\pi}$", fontsize=35)
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)
plt.grid(which='both')  # showing both major and minor grid lines

plt.yscale('log')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.grid(which='both')  # showing both major and minor grid lines

plt.tick_params(axis='y', which='minor')
ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.47), ncol=2)

plt.savefig('./results/creatinine_sqrt_rel_mse.pdf', bbox_inches='tight')


# DOF target = 5



