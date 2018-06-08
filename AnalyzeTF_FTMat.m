A=load('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/srez/RegridTry1C2_dataNeighborhoodR__2018-06-07_17-43-50_train/TrainSummary_028420.mat');
Flds=fieldnames(A);

X=A.gene_GEN_L004_M2D_MC_weightR_0+1i*A.gene_GEN_L004_M2D_MC_weightI_0;
Y=A.gene_GEN_L005_M2D_MC_weightR_0+1i*A.gene_GEN_L005_M2D_MC_weightI_0;
%%
fgmontage(fft1cg(X,2))
%%
fgmontage(fft1cg(Y,2))


%%

VP='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/srez/RegridTry1C2_TS2_dataNeighborhoodRCB0__2018-06-08_16-17-56_train/';
D=dir([VP 'Tra*.mat']);
Q=load([VP D(end).name]);
%%
Flds=fieldnames(Q);
SFlds=sort(Flds);
for i=1:numel(SFlds)
    disp([PadStringWithBlanks(SFlds{i},45) num2str(size(Q.(SFlds{i})),'% 9d         ')]);
end
%%
% size(NMapCR)
nTS=7;
PWK=double(Q.gene_GEN_L003_PixelswiseMult_weightR_0 + 1i*Q.gene_GEN_L003_PixelswiseMult_weightI_0);
PWK=reshape(PWK,[131 131 96 nTS]);
PWKB=squeeze(double(Q.gene_GEN_L003_PixelswiseMult_bias_0(:,:,:,:,1)+1i*Q.gene_GEN_L003_PixelswiseMult_bias_0(:,:,:,:,2)));
FTX=Q.gene_GEN_L004_M2D_MC_weightR_0+1i*Q.gene_GEN_L004_M2D_MC_weightI_0;
FTY=Q.gene_GEN_L005_M2D_MC_weightR_0+1i*Q.gene_GEN_L005_M2D_MC_weightI_0;

PWI=double(Q.gene_GEN_L006_PixelswiseMult_weightR_0 + 1i*Q.gene_GEN_L006_PixelswiseMult_weightI_0);
PWI=reshape(PWI,[100 100 nTS]);

PWIB=squeeze(double(Q.gene_GEN_L006_PixelswiseMult_bias_0(:,:,:,:,1)+1i*Q.gene_GEN_L006_PixelswiseMult_bias_0(:,:,:,:,2)));


MCDR=reshape(PWI,[100 100 nTS]);

Phi1=angle(MCDR);

Phi2=Phi1-Phi1(:,:,1);
Phi2=angle(exp(1i*Phi2));
% fgmontage(Phi2)
%% Run the net
nukData=ADataIsPy(:,:,SliI,13).';
nukData=nukData(:,3:end);
nukDataCC=MultMatTensor(sccmtx(:,1:ncc).',nukData);

DataV=Row(nukDataCC);
In=DataV(NMapCX);
In=permute(In,[2 1 3]);
AfterPWK=squeeze(sum(In.*PWK,3));
AfterPWKB=AfterPWK+PWKB;
% AfterFTX=MultMatTensor(FTX.',AfterPWKB);
AfterFTX=MultMatTensor(FTX.',permute(AfterPWKB,[2 1 3]));
AfterFTY=MultMatTensor(FTY.',permute(AfterFTX,[2 1 3]));
AfterPWI=sum(permute(PWI,[2 1 3]).*AfterFTY,3);
% AfterPWI=sum(PWI.*AfterFTY,3);
AfterPWIB=AfterPWI+PWIB;

ShowAbsAngle(AfterPWIB)

%%
M=load('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/OnRealData.mat');


MC=M.x(:,:,:,1)+1i*M.x(:,:,:,2);
MC=permute(MC,[2 3 1]);

%%
for r=1:nReps
    nukData=ADataIsPy(:,:,SliI,r).';
    nukData=nukData(:,3:end);
    nukDataCC=MultMatTensor(sccmtx(:,1:ncc).',nukData);

    CurIDataV=Row(nukDataCC.')*60;
    CurIDataVR=[real(CurIDataV) imag(CurIDataV)];
        
    Data=repmat(single(CurIDataVR),[16 1]);
    RealDataFN=['/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/srez/RealData/b_Ben14May_Sli5_r' num2str(r,'%02d') '.mat'];
    save(RealDataFN,'Data');
end
%%
for r=1:nReps
    M=load(['/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/Out/OnRealData' num2str(r,'%02d') '.mat']);

    MC=M.x(:,:,:,1)+1i*M.x(:,:,:,2);
    MC=permute(MC,[2 3 1]);
    
    AMC(:,:,r)=MC(:,:,1);
end
%%
MCE=abs(AMC(:,:,2:2:end));
MCO=abs(AMC(:,:,1:2:end));
D=MCE-MCO;
fgmontage(mean(D(:,:,10:20),3))
%%
fgmontage(mean(D(:,:,5:35),3))

