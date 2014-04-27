#! /usr/bin/env python

import numpy as np
import vtk
from vtk.util import numpy_support
import butils
import argparse
import os

d_scale = 1.0


def progress_renwin(renWin):
    fname = renWin.fnames.next()
    dyn = np.load(fname.strip())
    rp = butils.pad_to_3d(np.array([dyn['rp']]))
    renWin.points.SetData(numpy_support.numpy_to_vtk(rp))

    keys = ['vp', 'vh', 've']
    # Something to do with VTK messing up Python's garbage collection
    # not sure why this is needed but it works anyway
    vs = [None for i in range(len(keys))]
    for i in range(len(renWin.point_sets)):
        vs[i] = butils.pad_to_3d(np.array([dyn[keys[i]]]))
        renWin.point_sets[i].SetVectors(numpy_support.numpy_to_vtk(vs[i]))

    renWin.Render()
    return fname


def progress_iren(obj, *args, **kwargs):
    progress_renwin(obj.GetRenderWindow())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise porous states using VTK')
    parser.add_argument('dyns', nargs='*',
                        help='npz files containing dynamic states')
    parser.add_argument('-s', '--save', default=False, action='store_true',
                        help='Save plot')
    args = parser.parse_args()

    datdir = os.path.abspath(os.path.join(args.dyns[0], '../..'))

    dirname = os.path.join(
        os.path.dirname(os.path.commonprefix(args.dyns)), '..')
    stat = np.load(os.path.join(dirname, 'static.npz'))
    rcs, Rc, Rp, L = stat['rcs'], stat['Rc'], stat['Rp'], stat['L']
    rcs = butils.pad_to_3d(rcs)

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(500, 500)

    if args.save:
        renWin.OffScreenRenderingOn()
        winImFilt = vtk.vtkWindowToImageFilter()
        winImFilt.SetInput(renWin)
        writer = vtk.vtkJPEGWriter()
        writer.SetInputConnection(winImFilt.GetOutputPort())
    else:
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.Initialize()

    if np.isfinite(L):
        sys = vtk.vtkCubeSource()
        sys.SetXLength(L)
        sys.SetYLength(L)
        sys.SetZLength(L)
        sysMapper = vtk.vtkPolyDataMapper()
        sysMapper.SetInputConnection(sys.GetOutputPort())
        sysActor = vtk.vtkActor()
        sysActor.GetProperty().SetOpacity(0.2)
        sysActor.SetMapper(sysMapper)
        ren.AddActor(sysActor)

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(rcs))
    polypoints = vtk.vtkPolyData()
    polypoints.SetPoints(points)
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetThetaResolution(20)
    sphereSource.SetPhiResolution(20)
    sphereSource.SetRadius(Rc)
    env = vtk.vtkGlyph3D()
    env.SetSourceConnection(sphereSource.GetOutputPort())
    env.SetInputData(polypoints)
    envMapper = vtk.vtkPolyDataMapper()
    envActor = vtk.vtkActor()
    envMapper.SetInputConnection(env.GetOutputPort())
    envActor.SetMapper(envMapper)
    envActor.GetProperty().SetColor(0, 1, 0)
    envActor.GetProperty().SetOpacity(0.5)
    envActor.GetProperty().SetRepresentationToWireframe()
    ren.AddActor(envActor)

    particlePoints = vtk.vtkPoints()

    particlePolys = vtk.vtkPolyData()
    particlePolys.SetPoints(particlePoints)
    particles = vtk.vtkGlyph3D()
    particleSource = vtk.vtkSphereSource()
    particleSource.SetThetaResolution(30)
    particleSource.SetPhiResolution(30)
    particleSource.SetRadius(Rp)
    particles.SetSourceConnection(particleSource.GetOutputPort())
    particles.SetInputData(particlePolys)
    particlesMapper = vtk.vtkPolyDataMapper()
    particlesMapper.SetInputConnection(particles.GetOutputPort())
    particlesActor = vtk.vtkActor()
    particlesActor.SetMapper(particlesMapper)
    particlesActor.GetProperty().SetOpacity(0.5)
    ren.AddActor(particlesActor)

    dSource = vtk.vtkArrowSource()

    pPoint_sets, ps, pMappers, pActors = [], [], [], []
    for c in ((1, 0, 0), (0, 0, 1), (1, 1, 0)):
        polys = vtk.vtkPolyData()
        polys.SetPoints(particlePoints)
        p = vtk.vtkGlyph3D()
        p.SetSourceConnection(dSource.GetOutputPort())
        p.SetInputData(polys)
        p.SetScaleModeToScaleByVector()
        p.SetScaleFactor(d_scale)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(p.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*c)
        ren.AddActor(actor)

        pPoint_sets.append(polys.GetPointData())
        ps.append(p)
        pMappers.append(mapper)
        pActors.append(actor)

    renWin.fnames = iter(args.dyns)
    renWin.points = particlePoints
    renWin.point_sets = pPoint_sets

    if not args.save:
        iren.RemoveObservers('KeyPressEvent')
        iren.AddObserver('KeyPressEvent', progress_iren, 1.0)
        iren.Start()
    else:
        while True:
            fname = progress_renwin(renWin)
            print(fname)
            outname = os.path.splitext(fname)[0]
            winImFilt.Modified()
            writer.SetFileName('{}.jpg'.format(outname))
            writer.Write()
