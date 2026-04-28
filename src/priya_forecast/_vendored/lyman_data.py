"""Module to load the covariance matrix (from BOSS DR9 or SDSS DR5 data) from tables."""

import os.path
import pandas
import numpy as np
import numpy.testing as npt


class SDSSData:
    """A class to store the flux power and corresponding covariance matrix from SDSS. A little tricky because of the redshift binning."""

    def __init__(
        self, datafile="data/lya.sdss.table.txt", covarfile="data/lya.sdss.covar.txt"
    ):
        # Read SDSS best-fit data.
        # Contains the redshift wavenumber from SDSS
        # See 0405013 section 5.
        # First column is redshift
        # Second is k in (km/s)^-1
        # Third column is P_F(k)
        # Fourth column (ignored): square roots of the diagonal elements
        # of the covariance matrix. We use the full covariance matrix instead.
        # Fifth column (ignored): The amount of foreground noise power subtracted from each bin.
        # Sixth column (ignored): The amound of background power subtracted from each bin.
        # A metal contamination subtraction that McDonald does but we don't.
        cdir = os.path.dirname(__file__)
        datafile = os.path.join(cdir, datafile)
        covarfile = os.path.join(cdir, covarfile)
        data = np.loadtxt(datafile)
        self.redshifts = data[:, 0]
        self.kf = data[:, 1]
        self.pf = data[:, 2]
        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())
        assert self.nz * self.nk == np.size(self.kf)
        # The covariance matrix, correlating each k and z bin with every other.
        # kbins vary first, so that we have 11 bins with z=2.2, then 11 with z=2.4,etc.
        self.covar = np.loadtxt(covarfile)

    def get_kf(self, kf_bin_nums=None):
        """Get the (unique) flux k values"""
        kf_array = np.sort(np.array(list(set(self.kf))))
        if kf_bin_nums is None:
            return kf_array
        return kf_array[kf_bin_nums]

    def get_redshifts(self):
        """Get the (unique) redshift bins, sorted in decreasing redshift"""
        return np.sort(np.array(list(set(self.redshifts))))[::-1]

    def get_pf(self, zbin=None):
        """Get the power spectrum"""
        if zbin is None:
            return self.pf
        ii = np.where((self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01))
        return self.pf[ii]

    def get_icovar(self):
        """Get the inverse covariance matrix"""
        return np.linalg.inv(self.covar)

    def get_covar(self, zbin=None):
        """Get the covariance matrix"""
        _ = zbin
        return self.covar


class BOSSData(SDSSData):
    """A class to store the flux power and corresponding covariance matrix from BOSS."""

    def __init__(self, datafile=None, covardir=None):
        cdir = os.path.dirname(__file__)
        # by default load the more recent data, from DR14: Chanbanier 2019, arXiv:1812.03554
        if datafile is None or datafile == "dr14":
            datafile = os.path.join(cdir, "data/boss_dr14_data/Pk1D_data.dat")
            covarfile = os.path.join(cdir, "data/boss_dr14_data/Pk1D_cor.dat")
            systfile = os.path.join(cdir, "data/boss_dr14_data/Pk1D_syst.dat")
            # Read BOSS DR14 flux power data.
            # Fourth column: statistical uncertainty
            # Fifth and Sixth column (unused) are noise and side-band powers
            data = np.loadtxt(datafile)
            self.redshifts = data[:, 0]
            self.kf = data[:, 1]
            self.pf = data[:, 2]
            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())
            assert self.nz * self.nk == np.size(self.kf)
            # systematic uncertainies (8 contributions):
            # continuum, noise, resolution, SB, linemask, DLAmask, DLAcompleteness, BALcompleteness
            syst = np.loadtxt(systfile)
            self.covar_diag = np.sum(syst**2, axis=1) + data[:, 3] ** 2
            # The correlation matrix, correlating each k and z bin with every other.
            # file includes a series of 13 (for each z bin) 35x35 matrices
            corr = np.loadtxt(covarfile)
            self.covar = np.zeros(
                (len(self.redshifts), len(self.redshifts))
            )  # Full covariance matrix (35*13 x 35*13) for k, z
            for bb in range(self.nz):
                dd = corr[
                    35 * bb : 35 * (bb + 1)
                ]  # k-bin covariance matrix (35 x 35) for single redshift
                self.covar[35 * bb : 35 * (bb + 1), 35 * bb : 35 * (bb + 1)] = (
                    dd  # Filling in block matrices along diagonal
                )

        # load the older dataset, from DR9: Palanque-Delabrouille 2013, arXiv:1306.5896
        elif datafile == "dr9":
            datafile = os.path.join(cdir, "data/boss_dr9_data/table4a.dat")
            covardir = os.path.join(cdir, "data/boss_dr9_data")
            # Read BOSS DR9 flux power data. See Readme file.
            # Sixth column: statistical uncertainty
            # Ninth column: systematic uncertainty
            # correlation matrices for each redshift stored in separate files, "cct4b##.dat"
            data = np.loadtxt(datafile)
            self.redshifts = data[:, 2]
            self.kf = data[:, 3]
            self.pf = data[:, 4]
            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())
            assert self.nz * self.nk == np.size(self.kf)
            self.covar_diag = data[:, 5] ** 2 + data[:, 8] ** 2
            # The correlation matrix, correlating each k and z bin with every other.
            # kbins vary first, so that we have 11 bins with z=2.2, then 11 with z=2.4, etc.
            self.covar = np.zeros(
                (len(self.redshifts), len(self.redshifts))
            )  # Full matrix (35*12 x 35*12) for k, z
            for bb in range(self.nz):
                dfile = os.path.join(covardir, "cct4b" + str(bb + 1) + ".dat")
                dd = np.loadtxt(
                    dfile
                )  # k-bin correlation matrix (35 x 35) for single redshift
                self.covar[35 * bb : 35 * (bb + 1), 35 * bb : 35 * (bb + 1)] = (
                    dd  # Filling in block matrices along diagonal
                )
        else:
            raise NotImplementedError("SDSS Data %s not found!" % datafile)

    def get_covar(self, zbin=None):
        """Get the covariance matrix"""
        # Note, DR9 and DR14 datasets report correlation matrices,
        # hence the conversion factor (outer product of covar_diag)
        if zbin is None:
            # return the full covariance matrix (all redshifts) sorted in blocks from low to high redshift
            return self.covar * np.outer(
                np.sqrt(self.covar_diag), np.sqrt(self.covar_diag)
            )
        # return the covariance matrix for a specified redshift
        ii = np.where(
            (self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01)
        )  # Elements in full matrix for given z
        rr = (np.min(ii), np.max(ii) + 1)
        std_diag_single_z = np.sqrt(self.covar_diag[rr[0] : rr[1]])
        covar_matrix = self.covar[rr[0] : rr[1], rr[0] : rr[1]] * np.outer(
            std_diag_single_z, std_diag_single_z
        )
        npt.assert_allclose(
            np.diag(covar_matrix), self.covar_diag[rr[0] : rr[1]], atol=1.0e-16
        )
        return covar_matrix

    def get_covar_diag(self):
        """Get the diagonal of the covariance matrix"""
        return self.covar_diag


class KSData(SDSSData):
    """
    A class to store the flux power and corresponding covariance matrix from KODIAQ-SQUAD.

    Optional: Because KODIAQ-SQUAD data underestimate the uncertainty in the first four bins,
    which is estimated from the https://arxiv.org/pdf/2306.06316.pdf in Fig 11,
    So we also discard the first four bins here.

    Also, we make an optional argument to allow the covariance matrix to scale by a constant.
    """

    def __init__(self, datafile=None, conservative=True, scale_covar=1.0):
        cdir = os.path.dirname(__file__)
        # data from the supplementary material in Karacayli+21](https://academic.oup.com/mnras/article/509/2/2842/6425772)
        if conservative:
            datafile = os.path.join(
                cdir, "data/kodiaq_squad/final-conservative-p1d-karacayli_etal2021.txt"
            )
            # Read KODIAQ-SQUAD flux power data.
            # Column #1 : redshift
            # Column #2: k
            # Column #3: power
            # Column #4: estimated error
            a = pandas.read_csv(datafile, skiprows=[0], sep="|", header=None)
            self.redshifts = np.array(a[1], dtype="float")
            self.kf = np.array(a[2], dtype="float")
            self.pf = np.array(a[3], dtype="float")

            # self.covar_diag = np.array(a[4], dtype='float' ) # This is std not covariance ...
            # ... take the diag of the covariance matrix instead

            # Read KODIAQ-SQUAD covariance matrix data.
            covardatafile = os.path.join(
                cdir,
                "data/kodiaq_squad/final-conservative-covariance-karacayli_etal2021.txt",
            )
            # Covariance matrix data
            # shape (len of kf, len of kf) = (182, 182), this is across redshift bins [2.2, 4.6]
            self.covar = np.loadtxt(covardatafile)

            # Don't make it default, because we want to keep the original data and discard bins only when needed
            # Get rid of some bins to avoid underestimation of the error
            # self.discard_kbins()

            # Scale the covariance matrix by a constant
            self.covar *= scale_covar

            # Take the diag of the covariance ...
            self.covar_diag = np.diag(self.covar)

            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())

        else:
            datafile = os.path.join(
                cdir, "data/kodiaq_squad/detailed-p1d-results-karacayli_etal2021.txt"
            )
            covdatafile = os.path.join(
                cdir, "data/kodiaq_squad/ks-lya-stat-covariance-karacayli_etal_2021.txt"
            )
            covarsi4file = os.path.join(
                cdir, "data/kodiaq_squad/ks-si4-stat-covariance-karacayli_etal2021.txt"
            )

            # Read KODIAQ-SQUAD flux power data.
            # Column #1 : redshift
            # Column #2: k
            # Column #3: power
            # columns_to_use = [
            #     "z",
            #     "k",
            #     "P_ks",
            #     "P_xq",
            #     "P_metal",
            #     "P_fid",
            #     "estat_ks",
            #     "estat_xq",
            #     "estat_metal",
            #     "esyst_res_ks",
            #     "esyst_res_xq",
            #     "esyst_dla",
            #     "esyst_cont",
            #     "esyst_metal",
            # ]
            a = pandas.read_csv(
                datafile,
                sep="|",
                header=0,
            )
            self.kf = a["           k "].values
            self.redshifts = a["   z "].values

            # P1D needs to subtract metals here because they don't mask metals in the data
            # Quote: .... For both measurements, we also subtract the metal power and add its covariance first, so that the weights have all the statistical and systematic errors in place.
            self.pf = a["        P_ks "].values
            self.p_metals = a["      P_metal "].values
            self.pf -= self.p_metals

            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())
            # systematic uncertainies (8 contributions):
            # continuum, noise, resolution, DLA, metals
            self.covar_diag = np.zeros_like(self.kf)
            # error budget excepts the estat_ks and estat_metal, because they are the diags
            # of the covariance matrices
            error_budgets = [
                # "    estat_ks ", # diag of ks-lya-stat-covariance-karacayli_etal_2021.txt
                # " estat_metal ", # diag of ks-si4-stat-covariance-karacayli_etal2021.txt
                " esyst_res_ks ",
                "   esyst_dla ",
                "  esyst_cont ",
                " esyst_metal ",
            ]
            for col in error_budgets:
                self.covar_diag += a[col].values ** 2

            # Metals
            self.esyst_metal = a[" esyst_metal "].values ** 2
            self.covar_metal = np.loadtxt(covarsi4file)  # add the metal covariance

            # Add Covariance
            self.covar = np.loadtxt(covdatafile)
            self.covar += self.covar_metal

            # add the systematics error to the diagonal of covar
            diag_indices = np.diag_indices(self.covar.shape[0])
            self.covar[diag_indices] += self.covar_diag

            # Scale the covariance matrix by a constant
            self.covar *= scale_covar

            # Take the diag of the covariance ...
            self.covar_diag = np.diag(self.covar)

            # Metals
            self.covar_metal[diag_indices] += self.esyst_metal
            self.covar_metal *= scale_covar
            self.covar_diag_metal = np.diag(self.covar_metal)

    def discard_kbins(self, min_k: float = 0.015, max_k: float = 0.06):
        """
        The fourth bin is k = 0.0157527.
        """
        # Filter the k bins to remove problematic bins for the emulator
        ind = self.kf > min_k
        ind = ind & (self.kf < max_k)

        # Filter kf, pf by the index. This step discards the elements of kf and pf where kf <= 0.015.
        # The resulting kf and pf arrays will only contain elements that satisfy the condition.
        self.kf = self.kf[ind]
        self.pf = self.pf[ind]
        self.redshifts = self.redshifts[ind]

        # Store the indices of the filtered k bins for later use
        self.ind = ind

        # Apply the same index to filter the rows and columns of the covariance matrix.
        # This is crucial because the covariance matrix elements correspond to the variances and covariances
        # between the elements of kf. Since we've filtered kf and pf, we must apply the same filter to the
        # covariance matrix to maintain the correct correspondence between the data and their covariances.
        self.covar = self.covar[ind][:, ind]
        # Assertion to ensure the covariance matrix is square and its dimensions match the length of the filtered kf array
        assert self.covar.shape == (
            len(self.kf),
            len(self.kf),
        ), "Covariance matrix dimensions do not match the filtered kf array."
        print(
            f"LyaData: Discarded {np.sum(~ind)} k bins with k <= {min_k} or k >= {max_k}."
        )

        # Recalculate the number of redshift bins and k bins
        # Take the diag of the covariance ...
        self.covar_diag = np.diag(self.covar)

        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())

        # Metals
        self.p_metals = self.p_metals[ind]
        self.covar_metal = self.covar_metal[ind][:, ind]
        self.covar_diag_metal = np.diag(self.covar_metal)

    def get_covar_diag(self, zbin=None):
        """Get the diagonal of the covariance matrix"""
        ii = np.where((self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01))
        return self.covar_diag[ii]

    def get_covar_diag_metal(self, zbin=None):
        """Get the diagonal of the covariance matrix for metals"""
        ii = np.where((self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01))
        return self.covar_diag_metal[ii]

    def get_covar(self, zbin=None):
        """
        Get the covariance matrix
        available_z = np.array([2, 2.2, ... , 4.2])
        """
        if zbin is None:
            # return the full covariance matrix (all redshifts) sorted in blocks from low to high redshift
            return self.covar
        # return the covariance matrix for a specified redshift
        ii = np.where(
            (self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01)
        )  # Elements in full matrix for given z
        rr = (np.min(ii), np.max(ii) + 1)
        covar_matrix = self.covar[rr[0] : rr[1], rr[0] : rr[1]]
        return covar_matrix

    def get_covar_metal(self, zbin=None):
        """
        Get the covariance matrix for metals
        available_z = np.array([2, 2.2, ... , 4.2])
        """
        if zbin is None:
            # return the full covariance matrix (all redshifts) sorted in blocks from low to high redshift
            return self.covar_metal
        # return the covariance matrix for a specified redshift
        ii = np.where((self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01))
        rr = (np.min(ii), np.max(ii) + 1)
        covar_matrix = self.covar_metal[rr[0] : rr[1], rr[0] : rr[1]]
        return covar_matrix

    def get_p_metals(self, zbin=None):
        """
        Get the power spectrum for metals
        available_z = np.array([2, 2.2, ... , 4.2])
        """
        if zbin is None:
            return self.p_metals
        ii = np.where((self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01))
        return self.p_metals[ii]


class DESIEDRData(SDSSData):
    """A class to store the flux power and corresponding covariance matrix from DESI EDRP.
    Note that if datafile == fft, k P(k) / pi is returned. Wavenumbers are in 1/Angstrom.
    """

    def __init__(self, datafile="qmle"):
        cdir = os.path.dirname(__file__)
        if datafile == "qmle":
            # data from the supplementary material of https://arxiv.org/pdf/2306.06316.pdf
            # here https://zenodo.org/record/8007370
            datafile = os.path.join(
                cdir,
                "data/desi_edrp_qmle_data/desi-edrp-lyasb1subt-p1d-detailed-results.txt",
            )
            # Read DESI flux power data.
            # Column #1 : redshift
            # Column #4: k
            # Column #6: final power (note this matches figure 8 of the paper).
            # Column #-1: total estimated error
            a = np.loadtxt(datafile)
            self.redshifts = np.array(a[:, 0], dtype="float")
            self.kf = np.array(a[:, 3], dtype="float")
            self.pf = np.array(a[:, 6], dtype="float")
            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())

            #e_total
            # self.covar_diag = np.array(a[:,-1], dtype='float')
            #Covariance matrix files in the tarball:
            #This is the statistical covariance.
            #desi-edrp-lya-cov-stat.txt
            #This is very small, probably the statistical covariance in the sideband.
            #desi-edrp-sb1-cov-stat.txt
            #
            #These next two have the same diagonal, identical to the square of e_total in the power spectrum file.
            #This one has the off diagonal elements matching the statistical covariance file
            #desi-edrp-lyasb1subt-cov-total-results.txt
            #
            #This one has the extra correlated off-diagonal elements.
            #desi-edrp-lyasb1subt-cov-total-offdiag-results.txt
            self.covar = np.loadtxt("data/desi-edrp-lyasb1subt-cov-total-offdiag-results.txt")
            self.covar_diag = np.diag(self.covar)
        else:
            # data from the supplementary material of https://arxiv.org/pdf/2306.06311.pdf
            # here https://zenodo.org/record/8020269
            datafile = os.path.join(cdir, "data/desi_edrp_fft_data/p1d_measurement.txt")
            # Read DESI flux power data.
            # Column #1 : redshift
            # Column #4: k
            # Column #5: power
            # Column #-1: total estimated error
            a = np.loadtxt(datafile)
            self.redshifts = np.array(a[:, 0], dtype="float")
            self.kf = np.array(a[:, 1], dtype="float")
            self.pf = np.array(a[:, 2], dtype="float")
            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())
            self.covar_diag = np.array(a[:, 3], dtype="float")

    def get_covar(self, zbin=None):
        """Get the covariance matrix from DESI EDRP"""
        if zbin is None:
            # return the full covariance matrix (all redshifts) sorted in blocks from low to high redshift
            return self.covar
        # return the covariance matrix for a specified redshift
        ii = np.where((self.redshifts < zbin + 0.01)*(self.redshifts > zbin - 0.01)) #Elements in full matrix for given z
        rr = (np.min(ii), np.max(ii)+1)
        covar_matrix = self.covar[rr[0]:rr[1], rr[0]:rr[1]]
        return covar_matrix

    def get_covar_diag(self, zbin=None):
        """Get the diagonal of the covariance matrix"""
        ii = np.where((self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01))
        return self.covar_diag[ii]

    def get_kf(self, zbin=None):
        """Get the (unique) flux k values"""
        if zbin is None:
            return self.kf
        ii = np.where((self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01))
        return self.kf[ii]


class XQ100Data(SDSSData):
    """A class to store the flux power and corresponding covariance matrix from XQ100."""

    def __init__(self, datafile=None):
        cdir = os.path.dirname(__file__)

        # data from https://github.com/bayu-wilson/lyb_pk/tree/main/output , [Wilson+21](https://arxiv.org/abs/2106.04837)
        # This includes z = array([3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2])
        # But we only use z = array([3.4, 3.6, 3.8, 4. , 4.2])
        # TODO: check if this is true
        datafile = os.path.join(
            cdir, "data/xq100/pk_obs_corrNR_offset_DLATrue_metalTrue_res0.csv"
        )
        # Full Covariance, only has z = array([3.4, 3.6, 3.8, 4. , 4.2])
        covdatafile = os.path.join(
            cdir, "data/xq100/cov_xq100_lyb_corrected_onlyA_fromMCMC.txt"
        )

        # Read XQ100 flux power data.
        # Columns are labeled as 'z', 'k' and 'paa'

        a = pandas.read_csv(datafile)
        self.redshifts = np.array(a["z"], dtype="float")
        self.kf = np.array(a["k"], dtype="float")
        self.pf = np.array(a["paa"], dtype="float")

        # ========== Redshift Range 3.4 - 4.3 ==========
        # TODO: double check-
        # this covariance data only have
        available_z = np.array([3.4, 3.6, 3.8, 4.0, 4.2])

        # exclude zbins below 3.4
        ind = (self.redshifts >= available_z.min()) & (
            self.redshifts <= available_z.max()
        )
        self.redshifts = self.redshifts[ind]
        self.kf = self.kf[ind]
        self.pf = self.pf[ind]

        # ========== Covariance ==========
        data_cov = np.loadtxt(covdatafile)
        xind = data_cov[:, 0].astype("int")
        yind = data_cov[:, 1].astype("int")

        # empty covariance matrix and add values
        dim_i = int(np.sqrt(len(xind)))
        dim_j = int(np.sqrt(len(yind)))

        self.covar = np.zeros((dim_i, dim_j))

        # number of k bins is 13
        # here the cov is 65 by 65, where 65 = 13 * 5
        self.covar[xind, yind] = data_cov[:, 2]
        self.covar_diag = np.diag(self.covar)

        assert self.covar.shape[0] == self.kf.shape[0]
        assert self.covar.shape[1] == self.redshifts.shape[0]

        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())

    def discard_kbins(self, min_k: float = 0.015, max_k: float = 0.06):
        """
        The fourth bin is k = 0.0157527.
        """
        # Filter the k bins to remove problematic bins for the emulator
        ind = self.kf > min_k
        ind = ind & (self.kf < max_k)

        # Filter kf, pf by the index. This step discards the elements of kf and pf where kf <= 0.015.
        # The resulting kf and pf arrays will only contain elements that satisfy the condition.
        self.kf = self.kf[ind]
        self.pf = self.pf[ind]
        self.redshifts = self.redshifts[ind]

        # Store the indices of the filtered k bins for later use
        self.ind = ind

        # Apply the same index to filter the rows and columns of the covariance matrix.
        # This is crucial because the covariance matrix elements correspond to the variances and covariances
        # between the elements of kf. Since we've filtered kf and pf, we must apply the same filter to the
        # covariance matrix to maintain the correct correspondence between the data and their covariances.
        self.covar = self.covar[ind][:, ind]
        # Assertion to ensure the covariance matrix is square and its dimensions match the length of the filtered kf array
        assert self.covar.shape == (
            len(self.kf),
            len(self.kf),
        ), "Covariance matrix dimensions do not match the filtered kf array."
        print(
            f"LyaData: Discarded {np.sum(~ind)} k bins with k <= {min_k} or k >= {max_k}."
        )

        # Recalculate the number of redshift bins and k bins
        # Take the diag of the covariance ...
        self.covar_diag = np.diag(self.covar)

        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())

    def get_covar_diag(self, zbin=None):
        """
        Get the diagonal of the covariance matrix
        available_z = np.array([3.4, 3.6, 3.8, 4. , 4.2])
        """
        ii = np.where((self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01))
        return self.covar_diag[ii]

    def get_covar(self, zbin=None):
        """
        Get the covariance matrix
        available_z = np.array([3.4, 3.6, 3.8, 4. , 4.2])
        """
        if zbin is None:
            # return the full covariance matrix (all redshifts) sorted in blocks from low to high redshift
            return self.covar
        # return the covariance matrix for a specified redshift
        ii = np.where(
            (self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01)
        )  # Elements in full matrix for given z
        rr = (np.min(ii), np.max(ii) + 1)
        covar_matrix = self.covar[rr[0] : rr[1], rr[0] : rr[1]]
        return covar_matrix


class XQ1002017Data(SDSSData):
    """A class to store the flux power and corresponding covariance matrix from XQ100."""

    def __init__(self, datafile=None):
        cdir = os.path.dirname(__file__)

        # data from https://adlibitum.oats.inaf.it/XQ100survey/Data.html , https://arxiv.org/pdf/1702.01761
        # This includes z = array([3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2])
        datafile = os.path.join(cdir, "data/xq100_2017/pk_xs_final.txt")
        # Full Covariance, only has z = array([3, 3.2, 3.4, 3.6, 3.8, 4. , 4.2])
        covdatafile = os.path.join(cdir, "data/xq100_2017/cov_pk_xs_final.txt")

        # Read XQ100 flux power data.
        # Columns are labeled as 'z', 'k' and 'paa'

        a = np.loadtxt(datafile)
        self.redshifts = a[:, 0]
        self.kf = a[:, 1]
        self.pf = a[:, 2]

        # ========== Redshift Range 3.4 - 4.3 ==========
        # TODO: double check-
        # this covariance data only have
        available_z = np.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])

        # exclude zbins below 3.0
        ind = (self.redshifts >= available_z.min()) & (
            self.redshifts <= available_z.max()
        )
        self.redshifts = self.redshifts[ind]
        self.kf = self.kf[ind]
        self.pf = self.pf[ind]

        # ========== Covariance ==========
        # MNRAS 515, 857–870 (2022)
        # n order to correct
        # for the underestimation of the variance due to the limited number of
        # sightlines used (see e.g. Rollinde et al. 2012; Irˇ siˇ
        # c et al. 2013, 2017;
        # Wilson, Irˇ sic & McQuinn 2021), we multiplied the full matrix by a
        # factor of 1.3, which has been obtained by a comparison of different
        # methods for estimation of the covariance matrix.
        data_cov = np.loadtxt(covdatafile)
        xind = data_cov[:, 0].astype("int")
        yind = data_cov[:, 1].astype("int")

        # empty covariance matrix and add values
        dim_i = int(np.sqrt(len(xind)))
        dim_j = int(np.sqrt(len(yind)))

        self.covar = np.zeros((dim_i, dim_j))

        # number of k bins is 13
        # here the cov is 65 by 65, where 65 = 13 * 5
        self.covar[xind, yind] = data_cov[:, 2] * 1.3
        self.covar_diag = np.diag(self.covar)

        assert self.covar.shape[0] == self.kf.shape[0]
        assert self.covar.shape[1] == self.redshifts.shape[0]

        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())

    def discard_kbins(self, min_k: float = 0.015, max_k: float = 0.06):
        """
        The fourth bin is k = 0.0157527.
        """
        # Filter the k bins to remove problematic bins for the emulator
        ind = self.kf > min_k
        ind = ind & (self.kf < max_k)

        # Filter kf, pf by the index. This step discards the elements of kf and pf where kf <= 0.015.
        # The resulting kf and pf arrays will only contain elements that satisfy the condition.
        self.kf = self.kf[ind]
        self.pf = self.pf[ind]
        self.redshifts = self.redshifts[ind]

        # Store the indices of the filtered k bins for later use
        self.ind = ind

        # Apply the same index to filter the rows and columns of the covariance matrix.
        # This is crucial because the covariance matrix elements correspond to the variances and covariances
        # between the elements of kf. Since we've filtered kf and pf, we must apply the same filter to the
        # covariance matrix to maintain the correct correspondence between the data and their covariances.
        self.covar = self.covar[ind][:, ind]
        # Assertion to ensure the covariance matrix is square and its dimensions match the length of the filtered kf array
        assert self.covar.shape == (
            len(self.kf),
            len(self.kf),
        ), "Covariance matrix dimensions do not match the filtered kf array."
        print(
            f"LyaData: Discarded {np.sum(~ind)} k bins with k <= {min_k} or k >= {max_k}."
        )

        # Recalculate the number of redshift bins and k bins
        # Take the diag of the covariance ...
        self.covar_diag = np.diag(self.covar)

        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())

    def get_covar_diag(self, zbin=None):
        """
        Get the diagonal of the covariance matrix
        available_z = np.array([3.4, 3.6, 3.8, 4. , 4.2])
        """
        ii = np.where((self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01))
        return self.covar_diag[ii]

    def get_covar(self, zbin=None):
        """
        Get the covariance matrix
        available_z = np.array([3.4, 3.6, 3.8, 4. , 4.2])
        """
        if zbin is None:
            # return the full covariance matrix (all redshifts) sorted in blocks from low to high redshift
            return self.covar
        # return the covariance matrix for a specified redshift
        ii = np.where(
            (self.redshifts < zbin + 0.01) * (self.redshifts > zbin - 0.01)
        )  # Elements in full matrix for given z
        rr = (np.min(ii), np.max(ii) + 1)
        covar_matrix = self.covar[rr[0] : rr[1], rr[0] : rr[1]]
        return covar_matrix
