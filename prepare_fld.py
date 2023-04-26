# Routines to prepare FLD files for AAT observing. Takes a set of targets, standards, guide stars (fiducials) and sky fibre positions
# Splits and prioritises all the targets in each field, and then puts writes the FLD files with appropriate formatting

import numpy as np
import scipy as sp
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import date as dt
from astropy import units as u
from scipy.optimize import minimize
from astropy.coordinates import Angle

def format_sexagesimal(ra, dec):
	RA = Angle(ra, unit=u.deg)
	Dec = Angle(dec, unit=u.deg)
	replace_ra = False
	replace_dec = False
	if round(Dec.dms[2], 1) == 60.0:
		replace_dec = True
	if round(RA.hms[2], 2) == 60.00:
		replace_ra = True
	negate = False
	if Dec.dms[0] < 0:
		negate = True
		dec *= -1
		Dec = Angle(dec, unit=u.deg)
	return (
		f"{RA.hms[0]:02.0f} {RA.hms[1]:02.0f} {59.99 if replace_ra else f'{RA.hms[2]:05.2f}'} "
		f"{'-' if negate else '+'}{Dec.dms[0]:02.0f} "
		f"{Dec.dms[1]:02.0f} {59.9 if replace_dec else f'{Dec.dms[2]:04.1f}'}"
	)

def format_line(ID, coords, ttype, priority, mag, name, pmra=0, pmdec=0):
	return (
		f"{ID} {coords} {ttype} {priority} {mag:05.2f} 0 {pmra:.4f} {pmdec:.4f} {name}"
	)

# Reads in a set of targets, splits them into approriate lists and gives them priorities and tags
def get_field(field_name):

	inputfile = ""
	data = pd.read_csv(inputfile, delim_whitespace=True, escapechar="#")

	# Return objects data within the GW localisation region

	# Add PRIORITY to objects based on probability

	# Return dataframe containing targets
	return


# Reads in the guide, standard and sky fibre files, gives appropriate tag and priorities
def get_guides_standards_sky(field_name):

	# Read in the guide star file and isolate those in the localisation region
	guides = fits.open(str("./%s_guides_final.fit" % field_name))[1].data

	# Read in the standard star file and keep the brightest two in the localisation region
	standards = fits.open(str("./%s_standards_initial.fit" % field_name))[1].data
	standards.sort(order='Gmag')
	standards = standards[:2]

	# Read in the sky fibres file and get sky coordinates in the localisation region
	sky = pd.read_csv(str("./%s_sky.csv" % field_name), delim_whitespace=True)

	return guides, standards, sky

def write_fld(date, fieldname, coord, targets, guides, standards, sky):

	guide_min, guide_max = guides["Vmag"].min(), guides["Vmag"].max()

	header = (
		f"* Primary Target: VVV followup {fieldname}\n"
		f"*\n"
		f"* Anthony Carr, Cullan Howlett\n"
		f"* Last modified: {dt.today()}\n"
		f"*\n"
		f"* gals R <= 22.5 mag, guide stars {guide_min} < V < {guide_max}\n"
		f"LABEL Primary target {fieldname}\n"
		f"UTDATE {obs_date}\n"
		f"CENTRE {format_sexagesimal(coord[0], coord[1])}\n"
		f"EQUINOX J2000.0\n"
		f"PROPER_MOTIONS\n"
		f"* End of header info\n"
		f"* Start of input data\n"
		f"* TargetName(unique for header) RA(hh mm ss.ss) Dec([+/-]dd mm ss.s) TargetType(P,F,S) Priority(9 is highest) Magnitude(mm.mm) 0 pmRA pmDec TargetName"
	)
	strings = [header]

	# Main targets
	for i, row in targets.iterrows():
	    coords = format_sexagesimal(float(row[" RA(J2000)"]), float(row["Dec(J2000)"]))
	    strings.append(
	        format_line(
	            f"TG{i}",
	            coords,
	            "P",
	            int(row["PRIORITY"]),
	            row["V"],
	            "Program",
	        )
	    )

	# Standard stars
	for i, row in enumerate(standards):
	    coords = format_sexagesimal(float(row["RA_ICRS"]), float(row["DE_ICRS"]))
	    strings.append(
	        format_line(
	            row["WDJname"],
	            coords,
	            "P",
	            9,
	            float(row["Gmag"]),
	            "Program",
	            pmra=float(row["pmRA"] / 1000.0) / np.cos(float(row["DE_ICRS"])*np.pi/180.0),  # (RA)mas/yr -> "/yr
	            pmdec=float(row["pmDE"] / 1000.0),  # mas/yr -> "/yr
	        )
	    )

	# Guide stars
	for i, row in enumerate(guides):
	    coords = format_sexagesimal(float(row["RAJ2000"]), float(row["DEJ2000"]))
	    strings.append(
	        format_line(
	            row["UCAC4"],
	            coords,
	            "F",
	            5,
	            float(row["Vmag"]),
	            "guide",
	            pmra=float(row["pmRA"] / 1000.0) / np.cos(float(row["DEJ2000"])*np.pi/180.0),  # (RA)mas/yr -> "/yr
	            pmdec=float(row["pmDE"] / 1000.0),  # mas/yr -> "/yr
	        )
	    )

	# Sky fibres
	for i, row in sky.iterrows():
	    coords = format_sexagesimal(row["ra"], row["dec"])
	    strings.append(
	        format_line(
	            f"S{i}",
	            coords,
	            "S",
	            5,
	            23,
	            "sky",
	        )
	    )

	# Write out the FLD file
	with open(f"{fieldname}.fld", "w") as out:
	    for string in strings:
	        out.write(string)
	        out.write("\n")

	return

if __name__ == "__main__":

	obs_date = "2022 08 02"

	# A list of GW events we want to run this for
	fields = ["A", "B"] 

	# Lists containing the central coordinates of these fields
	coords = [[], []]

	# Loop over fields and central coordinates, get the data and write the fld file
	for field, coord in zip(fields, coords):
		targets = get_field(field)
		guides, standards, sky = get_guides_standards_sky(field)
		for i, targ in enumerate(targets):
			write_fld(obs_date, field, coord, targ, guides, standards, sky)



