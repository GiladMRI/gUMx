clear

run ( '~/HomeA/irt/setup.m');

cd('~/HomeA/gUM');
BasePX=[pwd filesep];
BaseP=[BasePX '4GL/'];

addpath([BaseP 'Spiral_recon_T1/'])
addpath(genpath([BaseP 'Spiral_recon_T1/raw_header']))
addpath([BaseP 'Spiral_recon_T1/misc'])
addpath([BaseP 'Spiral_recon_T1/io'])

addpath(genpath('~/HomeA/SPENOnline'))

addpath(genpath('~/HomeA/gUM/EPFLSpiral'))

addpath(genpath('~/HomeA/gpuNUFFT-master'))

addpath(genpath('~/HomeA/Tools/NIfTI_20140122'));

addpath('~/HomeA/gUM');

setenv('TOOLBOX_PATH','~/HomeA/bart-0.4.03')