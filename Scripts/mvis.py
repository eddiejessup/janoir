#! /usr/bin/env python

import numpy as np
import vtk
from vtk.util import numpy_support
import butils
import argparse
import os
import utils

d_scale = 0.3


def progress(obj, ev):
    fname = obj.fnames.next()
    dyn = np.load(fname.strip())
    rp = butils.pad_to_3d(np.array([dyn['rp']]))

    obj.points.SetData(numpy_support.numpy_to_vtk(rp))

    vp = butils.pad_to_3d(np.array([dyn['vp']]))
    obj.pPolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(vp))

    vh = butils.pad_to_3d(np.array([dyn['vh']]))
    obj.hPolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(vh))

    ve = butils.pad_to_3d(np.array([dyn['ve']]))
    obj.ePolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(ve))

    renWin.Render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise porous states using VTK')
    parser.add_argument('dyns', nargs='*',
                        help='npz files containing dynamic states')
    parser.add_argument('-s', '--save', default=False, action='store_true',
                        help='Save plot')
    args = parser.parse_args()

    datdir = os.path.abspath(os.path.join(args.dyns[0], '../..'))

    dirname = os.path.join(os.path.dirname(os.path.commonprefix(args.dyns)), '..')
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
    # dSource.SetTipResolution(1)
    # dSource.SetShaftResolution(1)

    pPolys = vtk.vtkPolyData()
    pPolys.SetPoints(particlePoints)
    ps = vtk.vtkGlyph3D()
    ps.SetSourceConnection(dSource.GetOutputPort())
    ps.SetInputData(pPolys)
    ps.SetScaleModeToScaleByVector()
    ps.SetScaleFactor(d_scale)
    pMapper = vtk.vtkPolyDataMapper()
    pMapper.SetInputConnection(ps.GetOutputPort())
    pActor = vtk.vtkActor()
    pActor.SetMapper(pMapper)
    pActor.GetProperty().SetColor(1, 0, 0)
    ren.AddActor(pActor)

    hPolys = vtk.vtkPolyData()
    hPolys.SetPoints(particlePoints)
    hs = vtk.vtkGlyph3D()
    hs.SetSourceConnection(dSource.GetOutputPort())
    hs.SetInputData(hPolys)
    hs.SetScaleModeToScaleByVector()
    hs.SetScaleFactor(d_scale)
    hMapper = vtk.vtkPolyDataMapper()
    hMapper.SetInputConnection(hs.GetOutputPort())
    hActor = vtk.vtkActor()
    hActor.SetMapper(hMapper)
    hActor.GetProperty().SetColor(0, 0, 1)
    ren.AddActor(hActor)

    ePolys = vtk.vtkPolyData()
    ePolys.SetPoints(particlePoints)
    es = vtk.vtkGlyph3D()
    es.SetSourceConnection(dSource.GetOutputPort())
    es.SetInputData(ePolys)
    es.SetScaleModeToScaleByVector()
    es.SetScaleFactor(d_scale)
    eMapper = vtk.vtkPolyDataMapper()
    eMapper.SetInputConnection(es.GetOutputPort())
    eActor = vtk.vtkActor()
    eActor.SetMapper(eMapper)
    eActor.GetProperty().SetColor(1, 1, 0)
    ren.AddActor(eActor)

    iren.fnames = iter(args.dyns)
    iren.points = particlePoints
    iren.pPolys = pPolys
    iren.ePolys = ePolys
    iren.hPolys = hPolys
    iren.RemoveObservers('KeyPressEvent')
    iren.AddObserver('KeyPressEvent', progress, 1.0)
    iren.Start()

    # for fname in args.dyns:
    #     dyn = np.load(fname.strip())
    #     rp = butils.pad_to_3d(np.array([dyn['rp']]))

    #     particlePoints.SetData(numpy_support.numpy_to_vtk(rp))

    #     vp = butils.pad_to_3d(np.array([dyn['vp']]))
    #     pPolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(vp))

    #     vh = butils.pad_to_3d(np.array([dyn['vh']]))
    #     hPolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(vh))

    #     ve = butils.pad_to_3d(np.array([dyn['ve']]))
    #     ePolys.GetPointData().SetVectors(numpy_support.numpy_to_vtk(ve))

    #     renWin.Render()

    #     if args.save:
    #         print(fname)
    #         outname = os.path.splitext(fname)[0]
    #         winImFilt.Modified()
    #         writer.SetFileName('{}.jpeg'.format(outname))
    #         writer.Write()

    # if not args.save:
    #     iren.Start()
