def rho_air(p, T, pv):
    """
    Calculate the air density in kg per m^3
    p is pressure (mb), T absolute temperature (K)
    pv partial pressure of water vapor (mb)
    """
    Rd = 287.058 #specific gas constant dry air
    Rv = 461.495 #specific gas constant water vapor            
    rho = (p-pv)/Rd+pv/Rv
    rho = 100*rho/T #since pressure is in mb (needed it in Pa)
    return rho

