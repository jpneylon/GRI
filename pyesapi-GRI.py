
# coding: utf-8

# In[ ]:


import json
import codecs
import pyesapi
import atexit


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, widgets


# In[ ]:


from scipy.interpolate import RegularGridInterpolator


# In[ ]:


app = pyesapi.CustomScriptExecutable.CreateApplication('pymoxie')
atexit.register(app.Dispose)


# In[ ]:


for pat_sum in app.PatientSummaries:
    print(pat_sum.Id, pat_sum.LastName, pat_sum.FirstName)


# In[ ]:


patid = input()


# In[ ]:


app.ClosePatient()


# In[ ]:


patient = app.OpenPatientById(patid)


# In[ ]:


[c.Id for c in patient.Courses]


# In[ ]:


cid = input()


# In[ ]:


course = patient.CoursesLot(cid)
print(course.Id)


# In[ ]:


[p.Id for p in patient.CoursesLot(cid).PlanSetups]


# In[ ]:


pid = input()


# In[ ]:


plan = patient.CoursesLot(cid).PlanSetupsLot(pid)


# In[ ]:


print(f'Plan Id: {plan.Id}')
print(f'Dose Per Fx: {plan.PrescribedDosePerFraction}')
print(f'Number of Fx: {plan.NumberOfFractions}')


# In[ ]:


import pandas as pd

df = pd.DataFrame(
    [(s.Id, s.DicomType, s.Volume, s.IsHighResolution) for s in plan.StructureSet.Structures],
    columns = ('StructureId', 'DicomType', 'Volume[cc]', 'IsHighRes')
)
df


# In[ ]:


constraint_filepath = "C:\\Users\\jneylon\\Documents\\JSON_Constraints\\" + patid + "-tight.json"


# In[ ]:


print(constraint_filepath)


# In[ ]:


json_data = json.load(codecs.open(constraint_filepath, 'r', 'utf-8-sig'))


# In[ ]:


json_data['CalculatedConstraints']


# In[ ]:


ct = plan.StructureSet.Image  # just a shortcut
ct_image = plan.StructureSet.Image.np_array_like()
print(ct.XSize, ct.YSize, ct.ZSize)
print(ct_image.shape)


# In[ ]:


rxdose = plan.TotalDose.Dose


# In[ ]:


dose = (rxdose / 100.) * plan.Dose.np_array_like(plan.StructureSet.Image)


# In[ ]:


gtv_structure = plan.StructureSet.StructuresLot('.GTV')


# In[ ]:


ptv_structure = plan.StructureSet.StructuresLot('.PTV')


# In[ ]:


slice_num = 180
plt.figure(figsize=(10,5))
plt.imshow(ct_image[200:400,200:400,slice_num].T, cmap='gray')

for pt_list in ptv_structure.GetContoursOnImagePlane(slice_num):
    plt.plot(
        [(pt.x-ct.Origin.x)/ct.XRes - 200 for pt in pt_list],
        [(pt.y-ct.Origin.y)/ct.YRes - 200 for pt in pt_list],
        'r'
    )
    
plt.axis('off')  # turn of axis numbers
plt.show()


# In[ ]:


def plot_contour_at_slice(structure, roi, style):
    for pt_list in structure.GetContoursOnImagePlane(roi[2]):
        plt.plot(
            [(pt.x-ct.Origin.x)/ct.XRes - roi[0].start for pt in pt_list],
            [(pt.y-ct.Origin.y)/ct.YRes - roi[1].start for pt in pt_list],
            style
        )


# In[ ]:


def plot_slice(slice_index,dose_level):
    plt.figure(figsize=(20,10))
    plt.imshow(ct_image[roi_slice[0],roi_slice[1],slice_index].T, cmap='gray', vmax=2000)
    
    dose_slice = dose[roi_slice[0],roi_slice[1],slice_index]
    dose_slice_mask = np.ma.masked_where(dose_slice<dose_level,dose_slice)
    plt.imshow(dose_slice_mask.T, cmap='jet', alpha=.3, vmax=rxdose)
    
    cb=plt.colorbar()
    cb.set_alpha(1.0)  # so colorbar is not transparent
    cb.draw_all()
    
    _roi = (roi_slice[0],roi_slice[1],slice_index)
    plot_contour_at_slice(plan.StructureSet.StructuresLot('.PTV'),_roi,'r')
    plot_contour_at_slice(plan.StructureSet.StructuresLot('SpinalCord'),_roi,'b')
    plot_contour_at_slice(plan.StructureSet.StructuresLot('Esophagus'),_roi,'g')
    
    plt.title(f'slice #{slice_index} dose [%]')
    plt.show()


# In[ ]:


slice_num = 180
roi_slice=(slice(215,365),slice(250,400),slice_num)

slice_slider = widgets.IntSlider(
    value=slice_num,
    min=0,
    max=ct_image.shape[2]-1,
    step=1,
    description='Slice Number',
    continuous_update=False,
)

dose_slider = widgets.IntSlider(
    value=50,
    min=0,
    max=110,
    step=1,
    description='Min Dose [%]',
    continuous_update=False,
)

interactive(
    plot_slice,
    slice_index=slice_slider,
    dose_level=dose_slider
)


# In[ ]:


sampling = [ct.XRes, ct.YRes, ct.ZRes]


# In[ ]:


gradx, grady, gradz = np.gradient(dose, ct.XRes, ct.YRes, ct.ZRes)


# In[ ]:


grad_slice=(slice(215,365),slice(250,400),slice_num)

plt.figure(figsize=(10,5))
plt.imshow(gradx[grad_slice].T,cmap='jet')
plt.colorbar()
plt.show()


# In[ ]:


grad_mag = np.sqrt(np.square(gradx) + np.square(grady) + np.square(gradz))


# In[ ]:


np.amax(grad_mag)


# In[ ]:


mag_slice=(slice(215,365),slice(250,400),slice_num)

plt.figure(figsize=(10,5))
plt.imshow(grad_mag[mag_slice].T,cmap='jet')
plt.colorbar()
plt.show()


# In[ ]:


evalid = int(input())


# In[ ]:


coneval = json_data['CalculatedConstraints'][evalid]


# In[ ]:


gUncertainty = float(input())


# In[ ]:


def plot_grad_slice(slice_index,grad_level):
    plt.figure(figsize=(10,5))
    plt.imshow(ct_image[g_slice[0],g_slice[1],slice_index].T, cmap='gray', vmax=2000)
    
    grad_slice = grad_mag[g_slice[0],g_slice[1],slice_index]
    grad_slice_mask = np.ma.masked_where(grad_slice<grad_level,grad_slice)
    plt.imshow(grad_slice_mask.T, cmap='jet', alpha=.3, vmax=np.amax(grad_mag))
    
    cb=plt.colorbar()
    cb.set_alpha(1.0)  # so colorbar is not transparent
    cb.draw_all()
    
    _roi = (roi_slice[0],roi_slice[1],slice_index)
    plot_contour_at_slice(plan.StructureSet.StructuresLot('.PTV'),_roi,'r')
    plot_contour_at_slice(plan.StructureSet.StructuresLot('SpinalCord'),_roi,'b')
    plot_contour_at_slice(plan.StructureSet.StructuresLot('Esophagus'),_roi,'g')
    
    plt.title(f'slice #{slice_index} dose [%]')
    plt.show()


# In[ ]:


g_slice=(slice(215,365),slice(250,400),slice_num)

grad_slider = widgets.IntSlider(
    value=10,
    min=0,
    max=np.amax(grad_mag),
    step=1,
    description='Min Gradient',
    continuous_update=False,
)

interactive(
    plot_grad_slice,
    slice_index=slice_slider,
    grad_level=grad_slider
)


# In[ ]:


coneval


# In[ ]:


structure = plan.StructureSet.StructuresLot(coneval['StructurePlan'])


# In[ ]:


structure.Volume


# In[ ]:


goal = float(coneval['GoalLimitToPresent'].replace(coneval['GoalLimitActualUnit'],''))


# In[ ]:


goal


# In[ ]:


achieved = float(coneval['Plan1ActualValue'])


# In[ ]:


achieved


# In[ ]:


def plot_dvh(structure):
    dvh = plan.GetDVHCumulativeData(
        structure,
        pyesapi.DoseValuePresentation.Absolute,
        pyesapi.VolumePresentation.Relative,
        .01
    )
    if dvh is not None:
        dose_x = [p.DoseValue.Dose for p in dvh.CurveData]
        volume_y = [p.Volume for p in dvh.CurveData]
        plt.plot(dose_x, volume_y, label=structure.Id)
    

plt.figure(figsize=(10,7))
plot_dvh(structure)
plt.legend(loc=0)
plt.title(f'Eclipse DVH: {plan.Id}')
plt.ylabel('Volume [%]')
plt.xlabel('Dose [cGy]')
plt.axvline(x=goal,color='g')
plt.axvline(x=achieved,color='r')
plt.show()


# In[ ]:


mask = ct.np_structure_mask(structure)
mask.shape


# In[ ]:


print(np.where(mask)[0][0], np.where(mask)[1][0], np.where(mask)[2][0])


# In[ ]:


mask_slice=(slice(215,365),slice(250,400),slice_num)

plt.figure(figsize=(10,5))
plt.imshow(mask[mask_slice].T,cmap='gray')
plt.colorbar()
plt.show()


# In[ ]:


x = np.linspace(0,ct.XSize-1,ct.XSize)
y = np.linspace(0,ct.YSize-1,ct.YSize)
z = np.linspace(0,ct.ZSize-1,ct.ZSize)


# In[ ]:


interpolater = RegularGridInterpolator((x,y,z), dose, method='linear', bounds_error='False')


# In[ ]:


dx = gUncertainty * np.divide(gradx, grad_mag, out=np.zeros_like(gradx), where=grad_mag!=0)


# In[ ]:


dy = -gUncertainty * np.divide(grady, grad_mag, out=np.zeros_like(grady), where=grad_mag!=0)


# In[ ]:


dz = gUncertainty * np.divide(gradz, grad_mag, out=np.zeros_like(gradz), where=grad_mag!=0)


# In[ ]:


thresh = 1000
q_slice = (slice(225,340),slice(280,370),172)
fig = plt.figure(figsize=(50,25))
plt.imshow(ct_image[q_slice].T, cmap='gray', vmax=2000)
plot_contour_at_slice(gtv_structure,q_slice,'w')
plot_contour_at_slice(structure,q_slice,'r')
#ax = fig.add_subplot(1,1,1)
#q = ax.quiver(X[q_slice].T, Y[q_slice].T, -dx[grad_slice], -dy[grad_slice], grad_mag[grad_slice], cmap='jet')
#plt.colorbar(q,extend='max')
plt.imshow(np.ma.masked_where(dose[q_slice]<thresh,dose[q_slice]).T, cmap='nipy_spectral', alpha=.3, vmax=2250.)
cb=plt.colorbar()
cb.set_alpha(1.0)  # so colorbar is not transparent
cb.draw_all()
plt.quiver(np.ma.masked_where(dose[q_slice]<thresh,gradx[q_slice]).T, np.ma.masked_where(dose[q_slice]<thresh,-grady[q_slice]).T)
plt.show()


# In[ ]:


mask_idx = np.nonzero(mask)


# In[ ]:


dose_array = interpolater(mask_idx)


# In[ ]:


mask_array = [mask_idx[0] + dx[mask_idx], mask_idx[1] + dy[mask_idx], mask_idx[2] + dz[mask_idx]]


# In[ ]:


moving_dose_array = interpolater(tuple(mask_array))


# In[ ]:


moving_dose = np.zeros_like(dose)
moving_dose[mask_idx] = moving_dose_array


# In[ ]:


def plot_moving_dose_slice(slice_index,dose_level):
    plt.figure(figsize=(20,10))
    plt.imshow(ct_image[m_slice[0],m_slice[1],slice_index].T, cmap='gray', vmax=2000)
    
    dose_slice = moving_dose[m_slice[0],m_slice[1],slice_index]
    dose_slice_mask = np.ma.masked_where(dose_slice<dose_level,dose_slice)
    plt.imshow(dose_slice_mask.T, cmap='nipy_spectral', alpha=.3, vmax=2250.)
    
    cb=plt.colorbar()
    cb.set_alpha(1.0)  # so colorbar is not transparent
    cb.draw_all()
    
    _roi = (m_slice[0],m_slice[1],slice_index)
    plot_contour_at_slice(gtv_structure,m_slice,'w')
    plot_contour_at_slice(structure,m_slice,'r')
    
    plt.title(f'slice #{slice_index} dose [cGy]')
    plt.show()


# In[ ]:


slice_num=172
m_slice=(slice(240,328),slice(290,355),slice_num)

move_slider = widgets.IntSlider(
    value=140,
    min=0,
    max=np.amax(moving_dose),
    step=1,
    description='Min Dose [cGy]',
    continuous_update=False,
)

interactive(
    plot_moving_dose_slice,
    slice_index=slice_slider,
    dose_level=move_slider
)


# In[ ]:


plt.figure(figsize=(20,15))

tot_vox = np.ones_like(dose)[mask_idx].sum()
hist,bins = np.histogram(moving_dose[mask_idx].flatten(),bins=1000,range=(0,moving_dose.max()))
        
plt.plot(bins[:-1],100.-hist.cumsum()*100.0/tot_vox,label=structure.Id)
dvh = plan.GetDVHCumulativeData(
        structure,
        pyesapi.DoseValuePresentation.Absolute,
        pyesapi.VolumePresentation.Relative,
        .01
    )

pts = np.array([[p.DoseValue.Dose,p.Volume] for p in dvh.CurveData])
plt.plot(pts[:,0],pts[:,1],'k--',alpha=.33)

plt.title("Moving Dose DVH (solid line) vs. Eclipse DVH (gray dashed line)",fontsize='xx-large')
plt.axvline(x=goal,color='g',label='Objective')
plt.axvline(x=achieved,color='r',label='Achieved')
plt.legend(loc=0,fontsize='xx-large')
plt.show()


# In[ ]:


dMargin = (goal - dose) * mask


# In[ ]:


dUncertainty = (moving_dose - dose) * mask


# In[ ]:


cord_slice=(slice(240,328),slice(290,355),172)
plt.figure(figsize=(20,10))
plt.imshow(ct_image[cord_slice].T, cmap='gray', vmax=2000)
plt.imshow(abs(moving_dose[cord_slice].T)*mask[cord_slice].T,alpha=0.5,cmap='nipy_spectral',vmin=1000.,vmax=2250.)
plot_contour_at_slice(gtv_structure,cord_slice,'w')
plot_contour_at_slice(structure,cord_slice,'r')
plt.colorbar()
plt.show()


# In[ ]:


np.amax(dUncertainty)


# In[ ]:


from skimage import measure


# In[ ]:


goal_percent = (goal / 2100.) * 100.


# In[ ]:


print(np.amax(dose),goal,goal_percent)


# In[ ]:


sub_slice=(slice(200,400),slice(200,400),slice(150,200))
sub_dose = dose[sub_slice]
sub_dose.shape


# In[ ]:


np.amax(sub_dose)


# In[ ]:


verts, faces, normals, values = measure.marching_cubes_lewiner(sub_dose,goal,gradient_direction='ascent',step_size=3)


# In[ ]:


from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# In[ ]:


fig = plt.figure(figsize=(50,50))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)
ax.set_xlim(0,200)
ax.set_ylim(0,200)
ax.set_zlim(0,50)
plt.tight_layout()
plt.show()


# In[ ]:


dose_mask = np.where(dose<goal,np.ones_like(dose),np.zeros_like(dose))
dose_mask.shape


# In[ ]:


dose_value = pyesapi.DoseValue(goal,pyesapi.DoseValue.DoseUnit.cGy)
print(dose_value)


# In[ ]:


dose_structure = pyesapi.Structure.ConvertDoseLevelToStructure(plan.Dose, goal, pyesapi.DoseValue.DoseUnit.cGy)


# In[ ]:


plt.figure(figsize=(20,10))
plt.imshow(dose_mask[q_slice].T,cmap='spectral')
plot_contour_at_slice(gtv_structure,q_slice,'w')
plot_contour_at_slice(structure,q_slice,'r')
plt.show()


# In[ ]:


from scipy.ndimage import morphology


# In[ ]:


sampling


# In[ ]:


dtdose = morphology.distance_transform_edt(dose_mask,sampling)


# In[ ]:


plt.figure(figsize=(20,10))
plt.imshow(dtdose[q_slice].T,cmap='nipy_spectral',vmax=20)
plot_contour_at_slice(gtv_structure,q_slice,'w')
plot_contour_at_slice(structure,q_slice,'r')
plt.colorbar()
plt.show()


# In[ ]:


gMargin = mask * dtdose


# In[ ]:


plt.figure(figsize=(20,10))
plt.imshow(gMargin[cord_slice].T,cmap='nipy_spectral', vmax=10)
plot_contour_at_slice(gtv_structure,cord_slice,'w')
plot_contour_at_slice(structure,cord_slice,'r')
plt.colorbar()
plt.show()


# In[ ]:


dRatio = np.divide(dUncertainty, dMargin, out=np.zeros_like(dMargin), where=dMargin!=0)


# In[ ]:


plt.figure(figsize=(20,10))
plt.imshow(abs(dRatio[cord_slice].T),cmap='nipy_spectral',vmax=3.)
plot_contour_at_slice(gtv_structure,cord_slice,'w')
plot_contour_at_slice(structure,cord_slice,'r')
plt.colorbar()
plt.show()


# In[ ]:


np.amax(dRatio)


# In[ ]:


gRatio = gUncertainty * np.divide(np.ones_like(gMargin), gMargin, out=np.zeros_like(gMargin), where=gMargin!=0)


# In[ ]:


np.amax(gRatio)


# In[ ]:


plt.figure(figsize=(20,10))
plt.imshow(gRatio[cord_slice].T,cmap='nipy_spectral',vmax=3.)
plot_contour_at_slice(gtv_structure,cord_slice,'w')
plot_contour_at_slice(structure,cord_slice,'r')
plt.colorbar()
plt.show()


# In[ ]:


GRI = np.sqrt(np.square(dRatio) + np.square(gRatio))


# In[ ]:


plt.figure(figsize=(20,10))
plt.imshow(GRI[cord_slice].T,cmap='nipy_spectral',vmin=0.5,vmax=3.)
plot_contour_at_slice(gtv_structure,cord_slice,'w')
plot_contour_at_slice(structure,cord_slice,'r')
plt.colorbar()
plt.show()


# In[ ]:


ind = np.unravel_index(np.argmax(GRI, axis=None), GRI.shape)
print(ind[2])


# In[ ]:


gri_slice=(slice(215,365),slice(250,400),ind[2])


# In[ ]:


plt.figure(figsize=(20,10))
plt.imshow(ct_image[cord_slice].T, cmap='gray', vmax=2000)
plot_contour_at_slice(gtv_structure,cord_slice,'w')
plot_contour_at_slice(structure,cord_slice,'r')
plt.imshow(np.ma.masked_where(GRI[cord_slice]<1.,GRI[cord_slice]).T, cmap='jet', alpha=.5, vmax=3.)
cb=plt.colorbar()
cb.set_alpha(1.0)  # so colorbar is not transparent
cb.draw_all()
plt.show()


# In[ ]:


np.amax(GRI)


# In[ ]:


def plot_gri(slice_index,risk_level):
    plt.figure(figsize=(20,10))
    plt.imshow(ct_image[r_slice[0],r_slice[1],slice_index].T, cmap='gray', vmax=2000)
    
    risk_slice = GRI[r_slice[0],r_slice[1],slice_index]
    risk_slice_mask = np.ma.masked_where(risk_slice<risk_level,risk_slice)
    plt.imshow(risk_slice_mask.T, cmap='nipy_spectral', alpha=.5, vmax=3)
    
    cb=plt.colorbar()
    cb.set_alpha(1.0)  # so colorbar is not transparent
    cb.draw_all()
    
    _roi = (roi_slice[0],roi_slice[1],slice_index)
    plot_contour_at_slice(plan.StructureSet.StructuresLot('.PTV'),_roi,'r')
    plot_contour_at_slice(plan.StructureSet.StructuresLot('SpinalCord'),_roi,'b')
    plot_contour_at_slice(plan.StructureSet.StructuresLot('Esophagus'),_roi,'g')
    
    plt.title(f'slice #{slice_index} Risk Index')
    plt.show()


# In[ ]:


r_slice=(slice(215,365),slice(250,400),slice_num)

risk_slider = widgets.IntSlider(
    value=1,
    min=0,
    max=3,
    step=1,
    description='GRI',
    continuous_update=False,
)

interactive(
    plot_gri,
    slice_index=slice_slider,
    risk_level=risk_slider
)


# In[ ]:


tot_vox = np.ones_like(GRI)[mask_idx].sum()
fail_idx = np.where(GRI>=1.0)
fail_vox = np.ones_like(GRI)[fail_idx].sum()
fail_pct = 100. * fail_vox / tot_vox


# In[ ]:


pd.options.display.float_format = '{:,.2f}'.format
df = pd.DataFrame(
    [(GRI[mask_idx].min(), GRI[mask_idx].max(), GRI[mask_idx].mean(), 100. * fail_vox / tot_vox)],
    columns = ('Minimum', 'Maximum', 'Mean', '% Failing Voxels')
)
df.transpose()

