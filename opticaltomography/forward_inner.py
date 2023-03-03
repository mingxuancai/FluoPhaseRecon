import numpy as np
import torch
import sys

dtype = torch.float32
np_dtype = np.float32
np_ctype = "complex64"
ctype = torch.complex64


class PhaseObject3D:
    """
    RI_obj:             refractive index of object(Optional) - 1.33
    RI:                 background refractive index (Optional) - 1.33
    voxel_size:         size of each voxel in (y,x,z), tuple
    slice_separation:   how far apart are slices separated
    """

    def __init__(self, shape, voxel_size, RI_obj=None, RI=1.0):
        assert len(shape) == 3, "shape should be 3 dimensional"
        self.RI_obj = RI * torch.ones(shape, dtype=dtype) if RI_obj is None else RI_obj.type(dtype)
        self.shape = shape
        self.RI = RI
        self.pixel_size = voxel_size[0]
        self.pixel_size_z = voxel_size[2]

        self.slice_separation = self.pixel_size_z * torch.ones((shape[2]-1,), dtype=dtype)

    def convertRItoPhaseContrast(self):
        self.contrast_obj = self.RI_obj - self.RI


class Multislice:
    """
    phase_obj_3d:               phase_obj_3d object defined from class PhaseObject3D
    fx_illu_list:               fluorescent source coordinate in x
    fy_illu_list:               fluorescent source coordinate in y
    fz_illu_list:               fluorescent source coordinate in z
    """
    def __init__(self, phase_obj_3d, fx_illu_list=[0], fy_illu_list=[0], fz_illu_list=[0], device='cpu', **kwargs):
        # kwargs: [wavelength, na, RI_measure, sigma, fx_illu_list, fy_illu_list, voxel_size, fz_illu_list, pad, pad_size]
        self.phase_obj_3d = phase_obj_3d
        self.shape = phase_obj_3d.shape
        self.wavelength = kwargs["wavelength"]
        self.na = kwargs["na"]
        self.RI = phase_obj_3d.RI
        self.ps_y = kwargs["voxel_size"][0]
        self.ps_x = kwargs["voxel_size"][1]
        self.ps_z = kwargs["voxel_size"][2]
        self.sigma = kwargs["sigma"]

        # illumination source coordinate
        assert len(fx_illu_list) == len(fy_illu_list), "fx dimension not equal to fy"
        self.fx_illu_list = fx_illu_list
        self.fy_illu_list = fy_illu_list
        self.fz_illu_list = fz_illu_list
        self.number_illum = len(self.fx_illu_list)  # source_num

        # torch.float32
        self.slice_separation = phase_obj_3d.slice_separation

        # setup sampling
        # datatype: complex64; numpy
        # unshifted
        self.xx, self.yy, self.uxx, self.uyy = self.sampling()
        self.xx_shifted = torch.fft.ifftshift(self.xx)
        self.yy_shifted = torch.fft.ifftshift(self.yy)
        self.uxx_shifted = torch.fft.ifftshift(self.uxx)
        self.uyy_shifted = torch.fft.ifftshift(self.uyy)

        # set pupil
        # datatype: torch.float32
        # ifftshifted
        self.pupil = self.genPupil(self.na, self.wavelength)

        # set propagation kernel
        # self.propkern = self.propKernel(prop_distance=self.ps_z).to(device)

        # set propagation kernel phase
        # ifftshifted
        self.fzlin = (self.RI/self.wavelength)**2 - self.uxx_shifted**2 - self.uyy_shifted**2
        self.fzlin[self.fzlin < 0] = 0
        self.fzlin = (self.fzlin)**0.5
        self.pupil_stop = (self.uxx_shifted**2 + self.uyy_shifted**2 <= torch.max(self.uxx_shifted)**2).type(dtype)
        # Fresnel Propagation Kernel
        self.prop_kernel_phase = 1.0j * 2.0 * np.pi * self.pupil_stop * self.fzlin

        # set initial_z_position
        # back propagate shape_z // 2
        self.initial_z_position = -1 * (self.shape[2]//2) * self.ps_z

        # set scattering model: not used currently
        # self._opticsmodel = {"MultiPhaseContrast": MultiPhaseContrast}
        # self.scat_model_args = kwargs

        self.test = []
        self.test_spherical = []
        self.test_final = []

    def setScatteringMethod(self, model="MultiPhaseContrast"):
        # Define Scattering method for tomography
        self.scat_model = model
        if model == "MultiPhaseContrast":
            if not hasattr(self.phase_obj_3d, 'contrast_obj'):
                self.phase_obj_3d.convertRItoPhaseContrast()
            self._x = self.phase_obj_3d.contrast_obj
        # self._scattering_obj = self._opticsmodel[model](self.phase_obj_3d, **self.scat_model_args)

    def forward(self, obj, device='cpu'):
        forward_scattered_predict = torch.zeros(self.number_illum, self.shape[0], self.shape[1])
        # print("1:{}".format(torch.cuda.memory_allocated(0)))

        for illu_idx in range(self.number_illum):  # number of emitting sources
            fx_source = self.fx_illu_list[illu_idx]
            fy_source = self.fy_illu_list[illu_idx]
            fz_source_layer = self.fz_illu_list[illu_idx]
            fields = self._forwardMeasure(fx_source, fy_source, fz_source_layer, obj=obj, device=device)

            # get only absolute value of fields
            forward_scattered_predict[illu_idx,:,:] = fields

        return forward_scattered_predict, fields

    def _forwardMeasure(self, fx_source, fy_source, fz_source, obj, device='cpu'):
        """
        From an inner emitting source, this function computes the exit wave.
        fx_source, fy_source, fz_source: source position in x, y, z
        obj: obj to be passed through
        """
        Nz = obj.shape[2]
        obj = obj.to(device)

        # MultiPhaseContrast, solves directly for the phase contrast
        # {i.e. Transmittance = exp(sigma * PhaseContrast)}
        self.transmittance = torch.exp(1.0j * self.sigma * obj).to(device)

        # Compute Forward: multislice propagation
        # u: ifftshifted; complex64; numpy
        u, _, _, fz_illu = self._genSphericalWave(fx_source, fy_source) # initial field
        # self.test_spherical.append(u)
        u = u.to(device)

        # Multislice
        for zz in range(Nz):

            u *= self.transmittance[:, :, zz]
            # print("1:{}".format(torch.cuda.memory_allocated(0)))

            if zz < obj.shape[2] - 1:
                u = self._propagationAngular_pad(u, self.slice_separation[zz], device=device)
                
            # print("2:{}".format(torch.cuda.memory_allocated(0)))
                # u = self.propkern.to(device) * u

        # Focus
        if obj.shape[2] > 1:
            u = self._propagationAngular_pad(u, torch.tensor(-1 * self.ps_z * ((Nz-1)/2)), device=device)
        
        # Microscope's Point Spread Function (estimate the pupil's defocus)
        # self.test_final.append(u)
        u = self._systemPupilconstraint(u, device=device)
        
        """
        u = torch.fft.fft2(u)
        u *= self.pupil.to(device)
        u = torch.fft.ifft2(u)
        """

        # Camera Intensity Measurement
        est_intensity = torch.abs(u)
        # est_intensity = torch.fft.ifftshift(est_intensity)
        est_intensity = est_intensity * est_intensity
        
        return est_intensity
    
    def _systemPupilconstraint(self, field, pupil=None, pad = 50, device='cpu'):
        self.pupil = self.pupil.to(device)
        pupil_pad = torch.zeros(self.shape[1]+2*pad, self.shape[0]+2*pad, dtype=torch.complex128)
        pupil_pad[pad:self.shape[1]+pad, pad:self.shape[0]+pad] = torch.fft.fftshift(self.pupil)
        pupil_pad = pupil_pad.to(device)
        
        field_pad = torch.zeros(self.shape[0]+2*pad, self.shape[1]+2*pad, dtype=torch.complex128)
        field_pad[pad:self.shape[1]+pad, pad:self.shape[0]+pad] = torch.fft.fftshift(field)
        field_pad = field_pad.to(device)
        
        field_pad = torch.fft.fft2(field_pad)
        field_pad *= pupil_pad
        field_pad = torch.fft.ifft2(field_pad)
        
        field_non_pad = field_pad[pad:self.shape[1]+pad, pad:self.shape[0]+pad]
        
        return field_non_pad.to(device)

    def _genSphericalWave(self, fx_source, fy_source, device='cpu', pad=50):
        # ifftshifted
        fx_source, fy_source = self._setIlluminationOnGrid(fx_source, fy_source)
        fz_source = self.RI/self.wavelength

        # spherical wave at z = 0
        r = ((self.xx_shifted-fx_source*self.shape[1])**2+(self.yy_shifted-fy_source*self.shape[0])**2+(self.ps_z*0)**2)**0.5

        r_nonzero = r
        r_nonzero[r_nonzero < 1e-5 ] = 1  # prevent divide by zero
        
        source_xy = torch.exp(1.0j * 2.0 * np.pi / self.wavelength * r_nonzero)/r_nonzero
        # set the zero value (center coordinate) to 10
        source_xy[source_xy == torch.exp(torch.tensor(1.0j * 2.0 * np.pi / self.wavelength))] = 1 # set to 1 now s 
        # source_xy[torch.abs(source_xy) > 10] = 1

        return source_xy.to(device), fx_source, fy_source, fz_source

    def _setIlluminationOnGrid(self, fx_source, fy_source, device='cpu'):
        fx_source_on_grid = np.round(fx_source*self.shape[1]/self.ps_x)*self.ps_x/self.shape[1]
        fy_source_on_grid = np.round(fy_source*self.shape[0]/self.ps_y)*self.ps_y/self.shape[0]

        return fx_source_on_grid, fy_source_on_grid

    def sampling(self, device='cpu'):
        # real space sampling
        y = (np.arange(self.shape[0]) - self.shape[0] // 2) * (self.ps_y)
        x = (np.arange(self.shape[1]) - self.shape[1] // 2) * (self.ps_x)
        xx, yy = np.meshgrid(x, y)
        xx = np.array(xx)
        yy = np.array(yy)

        # spatial frequency sampling
        uy = (np.arange(self.shape[0]) - self.shape[0] // 2) * (1 / self.ps_y / self.shape[0])
        ux = (np.arange(self.shape[1]) - self.shape[1] // 2) * (1 / self.ps_x / self.shape[1])
        uxx, uyy = np.meshgrid(ux, uy)
        uxx = np.array(uxx)
        uyy = np.array(uyy)

        return torch.from_numpy(xx).to(device), torch.from_numpy(yy).to(device), torch.from_numpy(uxx).to(device), torch.from_numpy(uyy).to(device)

    def genPupil(self, na, wavelength, device='cpu'):
        # pupil: shifted

        pupil_radius = na/wavelength
        pupil = (self.uxx**2 + self.uyy**2 <= pupil_radius**2)
        pupil_mask = (self.uxx**2 + self.uyy**2 <= torch.max(self.uxx)**2)
        pupil *= pupil_mask
        pupil = torch.fft.ifftshift(pupil)

        return pupil.to(device)

    def _propagationAngular_pad(self, field, prop_distance, test=False, device='cpu', pad = 50):
        """
        propagation operator that uses angular spectrum to propagate the wave

        field:                  input field: shifted
        propagation_distance:   distance to propagate the wave
        self.prop_kernel_phase: shifted
        """    
        
        self.prop_kernel_phase = self.prop_kernel_phase.to(device)
        prop_kernel_padding = torch.zeros(self.shape[0]+2*pad, self.shape[1]+2*pad, dtype=torch.complex128)
        prop_kernel_padding[pad:self.shape[0]+pad, pad:self.shape[0]+pad] = torch.fft.fftshift(self.prop_kernel_phase)
        prop_kernel_padding = prop_kernel_padding.to(device)
        
        field_pad = torch.zeros(self.shape[0]+2*pad, self.shape[1]+2*pad, dtype=torch.complex128)
        field_pad[pad:self.shape[0]+pad, pad:self.shape[0]+pad] = torch.fft.fftshift(field)
        field_pad = field_pad.to(device)
        
        field_pad = torch.fft.fft2(field_pad)
        if prop_distance > 0:
            field_pad[:, :] *= torch.exp(prop_kernel_padding * prop_distance)
            
        else:
            field_pad[:, :] *= torch.conj(torch.exp(prop_kernel_padding * torch.abs(prop_distance)))
        field_pad = torch.fft.ifft2(field_pad)
        
        field_non_pad = field_pad[pad:self.shape[0]+pad, pad:self.shape[0]+pad]
        field_non_pad = torch.fft.fftshift(field_non_pad)
        # print(field_non_pad.shape)
        
        return field_non_pad.to(device)
    
    def _propagationAngular(self, field, prop_distance, test=False, device='cpu'):
        """
        propagation operator that uses angular spectrum to propagate the wave

        field:                  input field: shifted
        propagation_distance:   distance to propagate the wave
        self.prop_kernel_phase: shifted
        """
        
        self.prop_kernel_phase = self.prop_kernel_phase.to(device)
        
        # if test:
            # self.test.append(field)
        field = torch.fft.fft2(field)
        
        # if test:
            # self.test.append(field)
        if prop_distance > 0:
            field[:, :] *= torch.exp(self.prop_kernel_phase * prop_distance)
            
        else:
            field[:, :] *= torch.conj(torch.exp(self.prop_kernel_phase * torch.abs(prop_distance)))

        field = torch.fft.ifft2(field)
        
        return field.to(device)







