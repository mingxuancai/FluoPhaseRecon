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
        # padded sampling
        # _, _, self.uxx_pad, self.uyy_pad = self.sampling(pad=self.shape[0])
        # self.uxx_shifted_pad = torch.fft.ifftshift(self.uxx_pad)
        # self.uyy_shifted_pad = torch.fft.ifftshift(self.uyy_pad)
        
        # set propagation kernel phase
        # ifftshifted
        self.fzlin = (self.RI/self.wavelength)**2 - self.uxx_shifted**2 - self.uyy_shifted**2
        self.fzlin[self.fzlin < 0] = 0
        self.fzlin = (self.fzlin)**0.5
        self.pupil_stop = (self.uxx_shifted**2 + self.uyy_shifted**2 <= torch.max(self.uxx_shifted)**2).type(dtype)
        # Fresnel Propagation Kernel
        self.prop_kernel_phase = 1.0j * 2.0 * np.pi * self.pupil_stop * self.fzlin
        # print(self.prop_kernel_phase.shape)
        
        # set pupil
        # datatype: torch.float32
        # ifftshifted
        self.pupil = self.genPupil(self.na, self.wavelength)
        # print(self.pupil.shape)

        # set initial_z_position
        # back propagate shape_z // 2
        self.initial_z_position = -1 * (self.shape[2]//2) * self.ps_z

        self.test = []
        self.test_spherical = []
        self.test_final = []
        self.test_prop = []

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
    
    def predictSLM(self, fx, fy, fz, obj, focal_length, device='cpu'):
        """
        Simulate the wave propagation through the reconstructed sample.
        Get the corresponding SLM pattern for phase conjugation.
        focal_length: should be negtive for back propagation
        """
        
        # Step 1: create source field for propagation and get the exit field
        fields = self._forwardMeasure_amp(fx, fy, fz, obj=obj, device=device)
        
        # Step 2: Phase conjugate the field and focus at the distance (-focal length)
        fields = self._propagationAngular(torch.conj(fields), torch.tensor(-focal_length), device=device)
        
        # Step 3: Fourier transform to obtain wavefront at the SLM plane
        fields = torch.fft.fft2(fields)
        
        # Step 4: System limitation
        fields *= self.pupil.to(device)
        
        fields = torch.fft.fftshift(fields)
        self.fields_test = fields
        
        # Step 5: Mapping to SLM
        SLM_field = torch.angle(fields).cpu().numpy()*(360/(2*np.pi))
        SLM_field[SLM_field==180.0] = 0 # phase unwrapping
        SLM_field = np.flip(SLM_field)
        
        return SLM_field

    def _forwardMeasure_amp(self, fx_source, fy_source, fz_source, obj, device='cpu'):
        """
        From an inner emitting source, this function computes the exit wave.
        fx_source, fy_source, fz_source: source position in x, y, z
        obj: obj to be passed through
        """
        obj = obj.to(device)

        self.transmittance = torch.exp(1.0j * self.sigma * obj).to(device)

        u, _, _, fz_illu = self._genSphericalWave(fx_source, fy_source, fz_source, device=device, prop_distance=self.ps_z*1) 

        u = u.to(device)
        
        Nz = obj.shape[2]
        # Multislice
        if fz_source!=0:
            Nz -= (np.ceil(fz_source) + 1)
            Nz = int(Nz)
        
        for zz in range(Nz):
            if fz_source!=0:
                zz = zz + np.ceil(fz_source) + 1
                zz = int(zz)
            u *= self.transmittance[:, :, zz]

            if zz < obj.shape[2] - 1:
                u = self._propagationAngular(u, self.slice_separation[zz], device=device)
        
        # Exit field
        est_amp = u
        
        return est_amp
        

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
        u, _, _, fz_illu = self._genSphericalWave(fx_source, fy_source, fz_source, device=device, prop_distance=self.ps_z*1) # initial field

        # self.test_spherical.append(u)
        u = u.to(device)
        
        # u = self._propagationAngular(u, 48 * self.slice_separation[0], test=True, device=device)

        # Multislice
        if fz_source != 0:
            Nz -= (np.ceil(fz_source))
            Nz = int(Nz)
            
        for zz in range(Nz):
            if fz_source != 0:
                zz = zz + np.ceil(fz_source)
                zz = int(zz)
            u *= self.transmittance[:, :, zz]

            if zz < obj.shape[2] - 1:
                u = self._propagationAngular(u, self.slice_separation[zz], device=device)

        # Refocus
        if obj.shape[2] > 1:
            u = self._propagationAngular(u, torch.tensor(-1 * self.ps_z * ((Nz-1)/2)), device=device)
        
        # System pupil
        u = torch.fft.fft2(u)
        u *= self.pupil.to(device)
        u = torch.fft.ifft2(u)
        
        # u = self._system_pupil_pad(field=u)
        
        # Camera Intensity Measurement
        est_intensity = torch.abs(u)
        est_intensity = est_intensity * est_intensity
        
        return est_intensity
    
    def _system_pupil_pad(self, field):
        pad = self.shape[0]//2
        pad_upper = pad+self.shape[0]
        
        field_pad = torch.zeros([self.shape[0]*2, self.shape[0]*2])
        field_pad[pad:pad_upper, pad:pad_upper] = field
        
        field_pad = torch.fft.fft2(field_pad)
        field_pad *= self.pupil
        field_pad = torch.fft.ifft2(field_pad)
        
        field = field_pad[pad:pad_upper, pad:pad_upper]
        
        return field

    def _genSphericalWave(self, fx_source, fy_source, fz_depth, prop_distance=0, device='cpu'):
        """
        need to be changed according to the depth
        """
        # ifftshifted
        # do not need to adjust the coordinate
        fx_source, fy_source = self._setIlluminationOnGrid(fx_source, fy_source)
        
        fz_source = self.RI/self.wavelength
        # fz_source = (self.RI/self.wavelength)**2 - self.uxx_shifted**2 - self.uyy_shifted**2
        
        # avoid the problem of divided by 0 when generating spherical wave
        # since we will not put the sources near the surface
        if fz_depth != 0:
            dz_prop_distance = self.ps_z + (np.ceil(fz_depth)-fz_depth) * self.ps_z # compute the spherical wave at the next slice (thin)
            r = ((self.xx_shifted-fx_source*self.shape[1])**2+(self.yy_shifted-fy_source*self.shape[0])**2+dz_prop_distance**2)**0.5     
    
        else:  # set the source at a certain distance from the bottom of the sample
            r = ((self.xx_shifted-fx_source*self.shape[1])**2+(self.yy_shifted-fy_source*self.shape[0])**2+prop_distance**2)**0.5
        
        source_xy = torch.exp(1.0j * 2.0 * np.pi / self.wavelength * r)/r

        return source_xy.to(device), fx_source, fy_source, fz_source

    def _setIlluminationOnGrid(self, fx_source, fy_source, device='cpu'):
        
        fx_source_on_grid = np.round(fx_source*self.shape[1]/self.ps_x)*self.ps_x/self.shape[1]
        fy_source_on_grid = np.round(fy_source*self.shape[0]/self.ps_y)*self.ps_y/self.shape[0]

        return fx_source_on_grid, fy_source_on_grid

    def sampling(self, pad=0, device='cpu'):
        # real space sampling
        shape_x = self.shape[1] + pad
        shape_y = self.shape[0] + pad
        y = (np.arange(shape_y) - shape_y // 2) * (self.ps_y)
        x = (np.arange(shape_x) - shape_x // 2) * (self.ps_x)
        xx, yy = np.meshgrid(x, y)
        xx = np.array(xx)
        yy = np.array(yy)

        # spatial frequency sampling
        uy = (np.arange(shape_y) - shape_y // 2) * (1 / self.ps_y / shape_y)
        ux = (np.arange(shape_x) - shape_x // 2) * (1 / self.ps_x / shape_x)
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
    
    def _propagationAngular(self, field, prop_distance, test=False, device='cpu'):
        """
        propagation operator that uses angular spectrum to propagate the wave

        field:                  input field: shifted
        propagation_distance:   distance to propagate the wave
        self.prop_kernel_phase: shifted
        """
        
        self.prop_kernel_phase = self.prop_kernel_phase.to(device)
        
        field = torch.fft.fft2(field)
        
        if prop_distance > 0:
            field[:, :] *= torch.exp(self.prop_kernel_phase * prop_distance)
            
        else:
            field[:, :] *= torch.conj(torch.exp(self.prop_kernel_phase * torch.abs(prop_distance)))

        field = torch.fft.ifft2(field)
        
        return field.to(device)

    def _propagationAngular_pad(self, field, prop_distance, device='cpu'):
        """
        propagation operator that uses angular spectrum to propagate the wave

        field:                  input field: shifted
        propagation_distance:   distance to propagate the wave
        self.prop_kernel_phase: shifted
        """
        pad = self.shape[0]//2
        pad_upper = pad+self.shape[0]
        
        self.prop_kernel_phase = self.prop_kernel_phase.to(device)
        
        field_pad = torch.zeros([self.shape[0]*2, self.shape[0]*2], dtype=torch.complex64)
        field_pad[pad:pad_upper, pad:pad_upper] = field
        
        # if test:
            # self.test.append(field)
        field_pad = torch.fft.fft2(field_pad)
        
        # if test:
            # self.test.append(field)
        if prop_distance > 0:
            field_pad[:, :] *= torch.exp(self.prop_kernel_phase * prop_distance)
            
        else:
            field_pad[:, :] *= torch.conj(torch.exp(self.prop_kernel_phase * torch.abs(prop_distance)))

        field_pad = torch.fft.ifft2(field_pad)
        field = field_pad[pad:pad_upper, pad:pad_upper]
        
        return field.to(device)






