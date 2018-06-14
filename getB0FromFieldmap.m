BaseP='/media/a/DATA/2018_01_25/';
FN='meas_MID131_dt_fieldmap_iso4mm_Ice_FID275';
FN='meas_MID131_dt_fieldmap_iso4mm_Ice_FID275';

BaseP='/media/a/DATA1/13May18/Phantom/';
FN='meas_MID377_BP_fieldmap_4echos_FID17766';

BaseP='/media/a/DATA1/13May18/Phantom/';
FN='meas_MID377_BP_fieldmap_4echos_FID17766';

BaseP='/media/a/DATA/13May18/Me/';
FN='meas_MID399_BP_fieldmap_4echos_FID17788';

% BaseP='/media/a/DATA/14May18/Ben/';
% FN='meas_MID123_BP_fieldmap_5echosX_FID17958';

sTwix = mapVBVD([BaseP FN '.dat'],'removeOS','ignoreSeg','doAverage');
Data=sTwix.image();

mkdir([BaseP FN]);
%%
nSlices=numel(sTwix.hdr.Phoenix.sSliceArray.asSlice);
% Ord=[1:2:nSlices 2:2:nSlices];
Ord=[2:2:nSlices 1:2:nSlices];
[~,ROrd]=sort(Ord);
%%
D=gpermute(Data,[1 3 4 2 8 5 6 7]);
PD=padLeft(D,24,2);
PD=PD(:,:,:,:,:,ROrd);
% PD=RepDotMult(PD,permute( mod(1:nSlices,2)*2-1,[1 6 3 4 5 2]));
I=squeeze(fft3cg(PD));

TEs_us=[sTwix.hdr.Phoenix.alTE{:}];
%%
% save([BaseP FN filesep 'IandsTwix.mat'],'I','sTwix');

%%
WhichTwo=[1 2];

M=squeeze(I(:,:,:,WhichTwo(1),:)./I(:,:,:,WhichTwo(2),:));
% Mag=abs(M);
Mag=squeeze(mean(abs(I(:,:,:,WhichTwo,:)),4));

Combined=squeeze(sum(Mag.*exp(1i*angle(M)),3));
% Combined=sum(Combined,5);
gammaMHz=42.5774806;
gammaHz=gammaMHz*1e6;

deltaTE_us=TEs_us(WhichTwo(2))-TEs_us(WhichTwo(1));
scanFreq_Hz=sTwix.hdr.Config.ScanFrequency; % SystemFrequency

dAngle=double(angle(Combined.*exp(1i*0*2*pi*scanFreq_Hz*deltaTE_us/1e6)));
B0_Hz=dAngle/(2*pi*deltaTE_us/1e6);

FirstEcho=squeeze(I(:,:,:,1,:));
Mg=grmss(FirstEcho,3);
%%
fgmontage(Mg);
gprint(get(gcf,'Number'),[BaseP FN filesep 'Mg'],[]) 
close(gcf);

save([BaseP FN filesep 'FirstEcho.mat'],'FirstEcho');
% load([BaseP FN filesep 'FirstEcho.mat'])
%%
WhichSli=1:nSlices;
% WhichSli=1:38;
% WhichSli=15:20;
[unwrapped] = cusackUnwrap(dAngle(:,:,WhichSli), Mg(:,:,WhichSli)/3000);
% fgmontage(unwrapped)

B0_HzU=unwrapped/(2*pi*deltaTE_us/1e6);
%%
fgmontage(B0_HzU,[-800 200]);colorbar

gprint(get(gcf,'Number'),[BaseP FN filesep 'B0_HzU'],[]) 
close(gcf);

save([BaseP FN filesep 'B0_HzU.mat'],'B0_HzU');
% load([BaseP FN filesep 'B0_HzU.mat'])
%%
SnsSzB=[128 128];
for SliI=1:nSlices
    disp([num2str(SliI) ' ' datestr(now)]); % 45 sec per slice!
    SensB(:,:,:,SliI)=RunESPIRiTForSensMaps(FirstEcho(:,:,:,SliI),0,SnsSzB);
end
%%
save([BaseP FN filesep 'Sens.mat'],'SensB');
for SliI=1:nSlices
    ShowAbsAngle(SensB(:,:,:,SliI))
    YLbl=['Sli' num2str(SliI,'%02d')];
    ylabel(YLbl);
    gprint(get(gcf,'Number'),[BaseP FN filesep 'Sens_' YLbl],[]) 
    close(gcf);
end
%%
load([BaseP FN filesep 'Sens.mat'])
SnsSzB=gsize(SensB,1:2);
%%
B0Q=imresizeBySlices(B0_HzU,SnsSzB);
%%
















%%
WhichSli=11;
for WhichSli=1:nSlices
    [PhiCostantini] = cunwrap(dAngle(:,:,WhichSli), struct('weight',Mg(:,:,WhichSli),'RoundK',false,'maxblocksize',60));
    DA=angle(gsum(Mg(:,:,WhichSli).*exp(1i*(PhiCostantini-dAngle(:,:,WhichSli)))));
    
    PhiCostantiniB=PhiCostantini-DA;
    L=round((PhiCostantiniB-dAngle(:,:,WhichSli))/(2*pi));
    UL=unique(L);
    clear Scr
    for i=1:numel(UL)
        B=L==UL(i);
        Curmg=Mg(:,:,WhichSli);
        Scr(i)=sum(Curmg(B));
    end
    [~,MI]=max(Scr);
    PhiCostantiniB=PhiCostantiniB-UL(MI)*2*pi;
    
    
%     PhiCostantiniM(:,:,WhichSli)=PhiCostantini;
    PhiCostantiniBM(:,:,WhichSli)=PhiCostantiniB;
    % fgmontage(angle(exp(1i*dAngle(:,:,WhichSli))),[-pi pi])
    % fgmontage(angle(exp(1i*PhiCostantiniB)),[-pi pi])
    %
    % fgmontage(PhiCostantiniB,[-2*pi 2*pi])
    % fgmontage(dAngle(:,:,WhichSli),[-2*pi 2*pi])
    B0_HzU(:,:,WhichSli)=PhiCostantiniB/(2*pi*deltaTE_us/1e6);
end
%%
fgmontage(PhiCostantini)
%%
% Cx=padRight(padLeft(Combined(:,:,WhichSli),5,2),6,2);
B0_HzUx=padRight(padLeft(B0_HzU,5,2),6,2);
B0Q=imresizeBySlices(B0_HzUx,Sz2);
B0Q=imresizeBySlices(B0_HzU,SnsSzB);
%%
for i=1:numel(sTwix.hdr.Phoenix.sSliceArray.asSlice)
    LocFieldMap(i)=sTwix.hdr.Phoenix.sSliceArray.asSlice{i}.sPosition.dTra;
end
%%
% SnsSz=[96 96];
% for SliI=1:nSlices
%     disp(SliI);
%     Sens(:,:,:,SliI)=RunESPIRiTForSensMaps(FirstEcho(:,:,:,SliI),0,SnsSz);
% end
% %%
% save([BaseP FN '.mat'],'I','sTwix','B0_Hz','Sens');
% 
% fgmontage(Mg);
% gprint(get(gcf,'Number'),[BaseP FN filesep 'Mg'],[]) 
% close(gcf);
%%

%%
WhichTwo=[1 2];

M=squeeze(I(:,:,:,WhichTwo(1),:)./I(:,:,:,WhichTwo(2),:));
Mag=squeeze(mean(abs(I(:,:,:,WhichTwo,:)),4));

Combined=squeeze(sum(Mag.*exp(1i*angle(M)),3));

deltaTE_us=TEs_us(WhichTwo(2))-TEs_us(WhichTwo(1));

dAngle=double(angle(Combined.*exp(1i*0*2*pi*scanFreq_Hz*deltaTE_us/1e6)));
B0_Hz=dAngle/(2*pi*deltaTE_us/1e6);
TwoPi_Hz=1e6/(deltaTE_us);
%%
WhichTwo=[2 3];

M=squeeze(I(:,:,:,WhichTwo(1),:)./I(:,:,:,WhichTwo(2),:));
Mag=squeeze(mean(abs(I(:,:,:,WhichTwo,:)),4));

Combined=squeeze(sum(Mag.*exp(1i*angle(M)),3));

deltaTE_us2=TEs_us(WhichTwo(2))-TEs_us(WhichTwo(1));

dAngle2=double(angle(Combined.*exp(1i*0*2*pi*scanFreq_Hz*deltaTE_us2/1e6)));
B0_Hz2=dAngle2/(2*pi*deltaTE_us2/1e6);
TwoPi_Hz2=1e6/(deltaTE_us2);
%%
WhichTwo=[3 4];

M=squeeze(I(:,:,:,WhichTwo(1),:)./I(:,:,:,WhichTwo(2),:));
Mag=squeeze(mean(abs(I(:,:,:,WhichTwo,:)),4));

Combined=squeeze(sum(Mag.*exp(1i*angle(M)),3));

deltaTE_us3=TEs_us(WhichTwo(2))-TEs_us(WhichTwo(1));

dAngle3=double(angle(Combined.*exp(1i*0*2*pi*scanFreq_Hz*deltaTE_us3/1e6)));
B0_Hz3=dAngle3/(2*pi*deltaTE_us3/1e6);
TwoPi_Hz3=1e6/(deltaTE_us3);
%%
WhichTwo=[1 3];

M=squeeze(I(:,:,:,WhichTwo(1),:)./I(:,:,:,WhichTwo(2),:));
Mag=squeeze(mean(abs(I(:,:,:,WhichTwo,:)),4));

Combined=squeeze(sum(Mag.*exp(1i*angle(M)),3));

deltaTE_us4=TEs_us(WhichTwo(2))-TEs_us(WhichTwo(1));

dAngle4=double(angle(Combined.*exp(1i*0*2*pi*scanFreq_Hz*deltaTE_us4/1e6)));
B0_Hz4=dAngle4/(2*pi*deltaTE_us4/1e6);
TwoPi_Hz4=1e6/(deltaTE_us4);
%%
ddeltaTE_us=deltaTE_us2-deltaTE_us;
ddAngle=angle(exp(1i*(dAngle2-dAngle+0*2*pi*scanFreq_Hz*ddeltaTE_us/1e6)));
B0_Hzx=ddAngle/(2*pi*ddeltaTE_us/1e6);