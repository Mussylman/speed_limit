import { createBrowserRouter } from 'react-router-dom'
import { Layout } from '../components/layout'
import {
  Dashboard,
  CamerasPage,
  CameraDetailPage,
  CameraFormPage,
  SmartCamerasPage,
  MapPage,
  ViolationsPage,
  ViolationDetailPage,
  AdminPage,
  AdminCamerasPage,
  VehicleDetailPage,
} from '../pages'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'cameras', element: <CamerasPage /> },
      { path: 'cameras/:id', element: <CameraDetailPage /> },
      { path: 'smart-cameras', element: <SmartCamerasPage /> },
      { path: 'map', element: <MapPage /> },
      { path: 'violations', element: <ViolationsPage /> },
      { path: 'violations/:id', element: <ViolationDetailPage /> },
      { path: 'admin', element: <AdminPage /> },
      { path: 'admin/cameras', element: <AdminCamerasPage /> },
      { path: 'admin/cameras/new', element: <CameraFormPage /> },
      { path: 'admin/cameras/:id/edit', element: <CameraFormPage /> },
      { path: 'vehicles/:plate', element: <VehicleDetailPage /> },
    ],
  },
])
