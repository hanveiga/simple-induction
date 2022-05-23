subroutine writefields_new(number,device)
  use ISO_C_BINDING
  use parameters_dg_2d
  implicit none
  type (c_ptr) :: device
  !!real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::host
  integer :: number
  character(len=10)::filename
  currentsol = 0
  call d2h(device,currentsol,nvar*nx*ny*m*m)
  write(filename,'(a, i5.5)') 'modes', number
  open(unit=8,status="REPLACE",file= trim(folder)//"/"//filename//".dat",&
       &form='UNFORMATTED',access='direct',recl=nx*ny*m*m*nvar*8)
  write(8,rec=1)currentsol
  close(8)
end subroutine writefields_new

subroutine writefields_cpu(number,field)
  use ISO_C_BINDING
  use parameters_dg_2d
  implicit none

  real(kind=8),dimension(1:m,1:m,1:ny,1:nx,1:nvar)::field
  integer :: number
  character(len=10)::filename
  write(filename,'(a, i5.5)') 'cpuno', number
  open(unit=8,status="REPLACE",file= trim(folder)//"/"//filename//".dat",&
       &form='UNFORMATTED',access='direct',recl=nx*ny*m*m*nvar*8)
  write(8,rec=1)field
  close(8)
end subroutine writefields_cpu


subroutine writefields(number,device,host)
  use ISO_C_BINDING
  use parameters_dg_2d
  implicit none
  type (c_ptr) :: device
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::host
  integer :: number
  character(len=10)::filename
  call d2h(device,host,nvar*nx*ny*m*m)
  write(filename,'(a, i5.5)') 'modes', number
  open(unit=8,status="REPLACE",file= trim(folder)//"/"//filename//".dat",&
       &form='UNFORMATTED',access='direct',recl=nx*ny*m*m*nvar*8)
  write(8,rec=1)host
  close(8)
end subroutine writefields

subroutine writedesc()
  use parameters_dg_2d
  implicit none
  call system('mkdir -p ' //folder )
  open(unit=10,status="REPLACE",file= trim(folder)//"/Descriptor.dat")
  write(10,*)nx,ny,m,nvar,boxlen_x,boxlen_y,gamma
  close(10)
end subroutine writedesc

subroutine writefields_diff(number,device,host,device2,host2,device3)
  use ISO_C_BINDING
  use parameters_dg_2d
  implicit none
  type (c_ptr) :: device, device2, device3
  real(kind=8),dimension(1:nx,1:ny,1:m,1:m,1:nvar)::host, host2
  integer :: number
  character(len=10)::filename
  call d2h(device,host,nvar*nx*ny*m*m)
  call h2d(u_eq,device2,nvar*nx*ny*m*m)
  call device_get_modes_from_nodes(device2,device3)
  call d2h(device3,host2,nvar*nx*ny*m*m)
  print*,u_eq(4,4,:,:,1)
  write(filename,'(a, i5.5)') 'modes', number
  open(unit=8,status="REPLACE",file= trim(folder)//"/"//filename//".dat",&
       &form='UNFORMATTED',access='direct',recl=nx*ny*m*m*nvar*8)
  write(8,rec=1)host - host2
  close(8)
end subroutine writefields_diff
