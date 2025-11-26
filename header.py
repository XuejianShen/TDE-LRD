import astropy.constants as con

G_pc_kms2_Msun = 4.301e-3
pc_per_kms_to_yr = con.pc.value/1000 / (365.25 * 24 * 3600)
c_kms = con.c.value/1000