"""
Copyright 2024 Soren Christensen
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (dicomutils),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import re
import pydicom
from pydicom.uid import UID as pdUID
import os
import numpy as np
import datetime
import SimpleITK as sitk

class DicomSeries:
    def __init__(self,seriesinstanceUID,slicetol=0.01,timedefinition="AcquisitionTime",is_enhanced=False):
        self.seriesinstanceUID=seriesinstanceUID
        self.timedefinition=timedefinition
        self.seriesdatetime = ''
        self.patientid= ''
        self.datasets=[]
        self.sortmatrix=np.zeros((1,1))    #this n,p matrix indicates slicelocation (n, low to high) and frame (p).
        self.stepsizes=-1
        self.equidistant=-1
        self.isophasic=False
        self.slicetol=slicetol
        self.nslices=-1
        self.nframes=-1
        self.seriesdescription=""
        self.modality=""
        self.issorted=False
        self.Zpos=1
        self.seriesdate_fallback_tostudy = True
        self.is_enhanced = is_enhanced



    def addDS(self,ds):
        self.datasets.append(ds)
        self.issorted=False
        
    def CalcProjZ(self,IOP,IPP):
        if (IOP==[] or IPP==[]):
            return 0
        IOPlist=IOP
        IPPlist=IPP
    
        pos=np.array([float(IPPlist[0]),float(IPPlist[1]),float(IPPlist[2])])
        rowcos=np.array([float(IOPlist[0]),float(IOPlist[1]),float(IOPlist[2])])
        colcos=np.array([float(IOPlist[3]),float(IOPlist[4]),float(IOPlist[5])])
        
        cp=np.cross(rowcos,colcos)
        return int(np.round(np.dot(cp,pos),2)*1000)

    def getSliceTime(self,ds):
        #TODO, a switch that can use content time, acqu time, midscan itme etc
        #if date is not there, then set it to 19000101. Sometimes this happen when deidentified
        #is time is not there, all set to 000000.  So there is no failure if the tag is not present right now
        date=datetime.datetime.strptime("19000101",'%Y%m%d')
        time=datetime.datetime.strptime("000000",'%H%M%S')
      
        
        if self.timedefinition=="AcquisitionTime":
            if "AcquisitionDate" in ds:
                try:
                    date=datetime.datetime.strptime(ds.data_element("AcquisitionDate").value,'%Y%m%d')
                except:
                    pass

            if "AcquisitionTime" in ds:
                try:
                    #ms or not?
                    if re.search("\.",ds.data_element("AcquisitionTime").value):
                        time=datetime.datetime.strptime(ds.data_element("AcquisitionTime").value,'%H%M%S.%f')
                    else:
                        time=datetime.datetime.strptime(ds.data_element("AcquisitionTime").value,'%H%M%S')

                except:
                    pass
                    
        
        

        elif self.timedefinition=="ContentTime":
            if "ContentDate" in ds:
                try:
                    date = datetime.datetime.strptime(ds.data_element("ContentDate").value, '%Y%m%d')
                except:
                    pass
            if "ContentTime" in ds:
                try:
                    # ms or not?
                    if re.search("\.", ds.data_element("ContentTime").value):
                        time = datetime.datetime.strptime(ds.data_element("ContentTime").value, '%H%M%S.%f')
                    else:
                        time = datetime.datetime.strptime(ds.data_element("ContentTime").value, '%H%M%S')

                except:
                    pass



        else:
            print("Default date and time applied")
                
        #now merge into a time stamp
        return datetime.datetime.combine(date.date(),time.time())

    def write_pixeldata_to_file_based_on_series(self,outputfolder,nparray,seriesdescription,seriesnumber,rescale_slope_intercept=None): #will write pixeldata to a file based on the series.
        assert(self.issorted)   
        #more checks to come here ....
        
        #
        #iterate slices in order
        for iframe in range(self.sortorder.shape[1]):
            for islice in  range(self.sortorder.shape[0]): 
                cindx=self.sortorder[islice,iframe]  
                cDS=self.datasets[cindx][0]
                newSOPInstanceUID=pydicom.uid.generate_uid()
                cDS.SOPInstanceUID=newSOPInstanceUID
                cDS.PixelData=nparray[:,:,islice,iframe].tostring()
                cDS.SeriesDescription=seriesdescription
                cDS.SeriesNumber=seriesnumber
                if rescale_slope_intercept:
                    cDS.RescaleSlope=rescale_slope_intercept[0] 
                    cDS.RescaleIntercept=rescale_slope_intercept[1] 
                cDS.save_as(outputfolder + '/' +newSOPInstanceUID+'.dcm' )

    def sort(self):     
        #create and array of 
        # zpos in micormeter (int), microseconds since start (int),instancenumber (int)
        sortcolumns=np.zeros( (len(self.datasets),4),int )
        k=0
        nslicestotal=len(self.datasets)
        slicetimes=[]
        for ds,location in self.datasets:
            #quick hack for enhanced MR. Longer term need a file->slice layer instead
            if self.is_enhanced:
                assert len(ds["PerFrameFunctionalGroupsSequence"].value) == 1
                IPP = ds["PerFrameFunctionalGroupsSequence"][0]["PlanePositionSequence"][0]["ImagePositionPatient"]
                IOP = ds["SharedFunctionalGroupsSequence"][0]["PlaneOrientationSequence"][0]["ImageOrientationPatient"]
            else:
                IOP=ds.ImageOrientationPatient
                IPP=ds.ImagePositionPatient

            InstanceNumber=ds.InstanceNumber
            zpos=self.CalcProjZ(IOP,IPP)
            slicetimes.append(self.getSliceTime(ds)) #uses member definition of time and translates to datetime (supports us resolution)
            sortcolumns[k,:]=[zpos,1,InstanceNumber,k]
            k=k+1
            
            
        lowesttime=min(slicetimes)  
        slicetimes_normalized=[(k-lowesttime).total_seconds()*1000000 for k in slicetimes]
        
        for k in range(len(slicetimes_normalized)):
             sortcolumns[k,1]=   slicetimes_normalized[k]
        
       
        #now lets sort this matrix from third column to first column  (InstanceNumber,Time,Position) - reversed
        sorted_indx = np.lexsort((sortcolumns[:,2],sortcolumns[:,1],sortcolumns[:,0]))    
        
        
        sortcolumns_sorted=sortcolumns[sorted_indx,:]
        
        
        uniqueZ=np.unique(sortcolumns_sorted[:,0])
        #now lets us summarize this as frame count per unique location
        counts=np.zeros((1,len(uniqueZ)),int)
        #print("Slice vs frame count")
        for k,zpos in enumerate(uniqueZ):
            counts[0,k]=sum(sortcolumns_sorted[:,0]==zpos)
            #print(f"{zpos:>10},  {zpos-uniqueZ[0]} {counts[0,k]}")



        unique_counts=np.unique(counts)

        if len(unique_counts)==1:
             self.isophasic=True    #unless it is..
             self.nframes=unique_counts[0]
             self.nslices=len(uniqueZ)
             self.timingmatrix = np.zeros((self.nslices, self.nframes), int)

             self.sortorder=np.zeros((self.nslices,self.nframes),int)
             
             for islice in range(self.nslices):
                 for iframe in range(self.nframes):
                     rowindx=iframe + self.nframes*islice
                     self.sortorder[islice,iframe]=sortcolumns_sorted[rowindx,3]
                     self.timingmatrix[islice,iframe]=sortcolumns_sorted[rowindx,1]
        else:
            self.isophasic=False   #default, not isophasic
            self.nframes=-1  #means NA
            self.nslices=len(uniqueZ)
            maxframes = unique_counts.max()
            self.timingmatrix = np.zeros((self.nslices, maxframes), int)*np.nan
            self.sortorder=sortcolumns_sorted[:,3:4]
            #special output to help work out which frames to remove
            maxframes=unique_counts.max()
            expected_frames=maxframes*self.nslices
            print(f"Expected frames {expected_frames}, have {nslicestotal}")
            #now assume increasing instancenumbers and calc which frames, files are incomplete
            inumbers=sortcolumns[:, 2]
            exclude_inumbers=[]
            for iframe in range(maxframes):
                #expected inumbers for this frame (assuming increasing inumbers per-frame
                expected_inumbers =  np.arange(iframe*self.nslices+1,(iframe+1)*self.nslices+1)
                for sl_index,expect in enumerate(expected_inumbers):
                    if not expect in inumbers:
                        print(f"Frame {iframe} not complete")
                        print(f"Instancenumber {expect} is missing - slice index {sl_index}")
                        expected_inumbers_present = [ inumber  for inumber in expected_inumbers.tolist() if inumber in inumbers]
                        exclude_inumbers=exclude_inumbers+expected_inumbers_present
                        break

            print("Quarantine these in order to quarantine frames with excluded i numbers:")
            for inumber in exclude_inumbers:
                row=np.argwhere(inumbers==inumber)[0][0]
                print(self.datasets[row][1])

            #Another strategy could be interpolation.
            #Get file names for all locations in matrix and fill in nan where missing
            expected_inumber_matrix=np.reshape(np.arange(1,expected_frames+1),[self.nslices,maxframes],order="F")
            file_matrix=np.empty((self.nslices,maxframes), dtype=object)

            for islice in range(self.nslices):
                for iframe in range(maxframes):
                    if expected_inumber_matrix[islice,iframe] in inumbers: #do we have it?
                        inumber_row = sortcolumns_sorted[:,2]==expected_inumber_matrix[islice,iframe]
                        dataset_indx = sortcolumns_sorted[inumber_row,3][0]
                        file_matrix[islice,iframe] = self.datasets[dataset_indx][1]
                    else:
                        file_matrix[islice, iframe] = "NA"

            #print(file_matrix)
            for islice in range(self.nslices):
                for iframe in range(maxframes):
                    if file_matrix[islice, iframe]=="NA":  # do we have it?
                        print(f"Slice {islice} frame {iframe} is missing. Expected instancenumber {iframe*self.nslices+islice+1}")
                        if iframe>0:
                            prev_frame = f"Previous frame is {iframe-1} with filename {file_matrix[islice, iframe-1]}"
                        else:
                            prev_frame = f"No previous frame"

                        if iframe < maxframes-1:
                            next_frame = f"Next frame is {iframe + 1} with filename {file_matrix[islice, iframe + 1]}"
                        else:
                            next_frame = f"No next frame"

                        print(prev_frame)
                        print(next_frame)

        self.Zpos = uniqueZ
        #equidistance?
        second_zpos_diff=np.unique(np.abs(np.diff(np.diff(uniqueZ))))  #expected to be 0, but this shows any aberations
        if np.any(second_zpos_diff>(self.slicetol*1000)):  #are ANY of these aberations outside of TOL?
            self.equidistant = 0
            #print("warning, not equidistant with tolerance {}. Max diff is {}".format(str(self.slicetol),np.max(second_zpos_diff / 1000)))
        else:
            self.equidistant = 1

             
        firstDS= self.datasets[self.sortorder[0,0]][0]
        if "SeriesDescription" in firstDS:
            self.seriesdescription=firstDS.data_element("SeriesDescription").value     
        
        if "Modality" in firstDS:
            self.modality=firstDS.data_element("Modality").value     
        else:
            self.modality="NA"
        
        self.patientid=firstDS.data_element("PatientID").value

        if "SeriesDate" in firstDS and len(firstDS["SeriesDate"].value)>0:
            date = firstDS.data_element("SeriesDate").value
        elif self.seriesdate_fallback_tostudy:
            if "StudyDate" in firstDS  and len(firstDS["StudyDate"].value)>0:
                date = firstDS.data_element("StudyDate").value
            else:
                date = None
        else:
            date= None

        if "SeriesTime" in firstDS  and len(firstDS["SeriesTime"].value)>0:
            time=firstDS.data_element("SeriesTime").value
            time = time.replace(":","")

            if time == "0":
                print("Special cfed fix")
                time = "000000"


            try:
                self.seriesdatetime = datetime.datetime.strptime(date+time[0:6],'%Y%m%d%H%M%S')
            except Exception as e:
                print(f"failure passing date in {self.datasets[self.sortorder[0,0]][1]}")
                raise(e)
        elif self.seriesdate_fallback_tostudy:
            if "StudyTime" in firstDS  and len(firstDS["StudyTime"].value)>0:
                time = firstDS.data_element("StudyTime").value
                self.seriesdatetime = datetime.datetime.strptime(date + time[0:6], '%Y%m%d%H%M%S')
            else:
                time = None
        else:
            self.seriesdatetime = None



        #stepsizes
        if "PixelSpacing" in firstDS:
            stepsizes = firstDS.PixelSpacing
            stepsizes.append(np.median(np.diff(uniqueZ))/1000   ) 
            self.stepsizes=np.array( [np.float32(stepsizes[0]),np.float32(stepsizes[1]),np.float32(stepsizes[2])] )
        else:
            self.stepsizes ="NA"
        
        self.issorted=True
    
    def __str__(self):
        if self.issorted==False:
            return
        
        myop=(self.seriesinstanceUID +":\n"
              "SeriesDescription: " + self.seriesdescription +"\n"
              "Slices: " + str(self.nslices) +"\n"
              "Frames: " + str(self.nframes) +"\n"
              "Isophasic: " + str(self.isophasic) +"\n"
              "Equidistant: " + str(self.equidistant))

        return myop
    
    def get_file_order(self):
        if not self.issorted:
            return [[]]
        
        #make a 2D list of files to load for this volume
        filelist=[]
        

        for d1 in range(self.sortorder.shape[0]):
            framelist=[]
            for d2 in range(self.sortorder.shape[1]):   
                cindx=self.sortorder[d1,d2]
                framelist.append(self.datasets[cindx][1])       
            filelist.append(framelist) 
        return filelist

    def sitkimage(self,use_rescale=False) -> (sitk.Image, np.ndarray):
        #TODO - make 4D capable
        files=self.get_file_order()

        img1=self.datasets[0][0].pixel_array

        if use_rescale:
            datamatrix = np.zeros((img1.shape[0], img1.shape[1], self.nslices, self.nframes), np.float32)
        else:
            myformat = img1.dtype
            if img1.dtype == ">u2":
                myformat = "u2"
            datamatrix=np.zeros( (img1.shape[0],img1.shape[1],self.nslices,self.nframes),myformat)

        for islice in range(self.nslices):
            for iframe in range(self.nframes):
                if use_rescale:
                    if "RescaleSlope" in self.datasets[self.sortorder[islice,iframe] ][0]:
                        slope = float(self.datasets[self.sortorder[islice,iframe] ][0].RescaleSlope)
                        intercept = float(self.datasets [ self.sortorder[islice,iframe] ][0].RescaleIntercept)
                        datamatrix[:, :, islice, iframe] = self.datasets[self.sortorder[islice, iframe]][0].pixel_array*slope + intercept
                    else:
                        print("Rescale slope not present - ignoring...")
                        datamatrix[:,:,islice,iframe]=self.datasets[self.sortorder[islice,iframe]][0].pixel_array
                else:
                    datamatrix[:,:,islice,iframe]=self.datasets[self.sortorder[islice,iframe]][0].pixel_array


        #use reader to construct a header
        flist=[f[0] for f in files]
        reader=sitk.ImageSeriesReader()

        reader.SetFileNames(flist)
        headerimg = reader.Execute()

        if self.nframes>1:
            #headerimg = reader.Execute()
            datamatrix_reordered=np.transpose(datamatrix,[2,0,1,3])
            dymimg=sitk.GetImageFromArray(datamatrix_reordered,isVector=True)
            dymimg.CopyInformation(headerimg)
            headerimg=dymimg

        return headerimg,datamatrix

    
    
def dicomscan(inputspec,series_grouping_tags="SeriesInstanceUID",slicetolin=0.01,timedefinition="AcquisitionTime",exclude_folder_name_regex=None):
    foundseries={}
    if type(inputspec) is str: #string input it searched recursively
        rootdir=inputspec
        #print(" scanning " + rootdir)
        
        for folder, subs, files in os.walk(rootdir):
            if exclude_folder_name_regex is not None:
                if re.search(exclude_folder_name_regex,folder):
                    continue
            for filename in files:
                #print filename
                fname=os.path.join(folder, filename)

                try:
                    print(fname)
                    ds = pydicom.read_file(fname)
                except Exception as excp:
                    print(f"Failed on reading {fname}")
                    raise(excp)

                #SOP CLass UID
                is_enhanced = False
                SOPclasstxt = pdUID(ds["SOPClassUID"].value).name
                if re.match("Enhanced",SOPclasstxt):
                    is_enhanced = True
                #generate UID
                UID=''
                for k in series_grouping_tags.split(","):
                    if k in ds:
                        UID=UID+ds.data_element(k).value+","
                    else:
                        assert(0) # we should not accept UID seperation by things that are not there
                UID=UID[0:-1]  #chop comma
                        
                if not UID in foundseries:

                    foundseries[UID] =  DicomSeries(UID,slicetol=slicetolin,timedefinition=timedefinition,is_enhanced=is_enhanced)
                #TODO - could have layer here dealing with enhanced splitting
                foundseries[UID].addDS( (ds,fname) )

    elif type(inputspec) is list:
        #we assume files

        for fname in inputspec:
            ds = pydicom.read_file(fname)
            UID=''
            for k in series_grouping_tags.split(","):
                if k in ds:
                    UID=UID+str(ds.data_element(k).value)+","
                else:
                    assert(0) # we should not accept UID seperation by things that are not there
            UID=UID[0:-1]  #chop comma
                    
            if not UID in foundseries:
                foundseries[UID] =  DicomSeries(UID,slicetol=slicetolin) 
            
            foundseries[UID].addDS( (ds,fname) )    
    
                       
            
            
        
    for seriesUID in foundseries.keys():
        #print "sorting " + seriesUID + " " + foundseries[  seriesUID ].datasets[0][1]
        foundseries[  seriesUID ].sort()  
             
        #now sort all the info into series structures
    return foundseries


def writeRGBdicom(fileloc,matrix,header):
    pass


def writeMRdicom(fileloc,matrix,header,bvalue = None):
    # header info by IE (PS.3.3)
    SOPInstanceUID= pydicom.uid.generate_uid()
    file_meta = pydicom.dataset.Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    file_meta.MediaStorageSOPInstanceUID = SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    #file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.FileMetaInformationGroupLength=10
    file_meta.FileMetaInformationGroupLength=file_meta.__sizeof__()
    ds = pydicom.dataset.FileDataset("dummy.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128,is_implicit_VR=False)






    # ds.is_little_endian = True
    # ds.is_implicit_VR = True

    # all are tpye 1 (mandatory) or type 2 (zero length if unknown otherwise actual value)
    # Patient

    ds.Modality = "MR"

    for key,val in header.items():
        #print(key)
        if key == "ImagePositionPatient":
            val = list(val)
        if not key in ds:
            ds.add_new(key,pydicom.dataset.dictionary_VR(key),val)

    ds.PatientPosition = "HFS"  # 2c but needed unless patient orientation code sequence is there

    # Frame of Reference
    #ds.FrameOfReferenceUID =  header["FrameOfReferenceUID"]
    #ds.PositionReferenceIndicator = ""
    # Equipment

    # Image
    ds.InstanceNumber = header["InstanceNumber"]
    if "AcquisitionDate" in header:
        ds.AcquisitionDate = header["AcquisitionDate"]

    if "AcquisitionTime" in header:
        ds.AcquisitionTime =  header["AcquisitionTime"]

    # Image Plane
    #ds.PixelSpacing =header["PixelSpacing"]
    #ds.ImageOrientationPatient = header["ImageOrientationPatient"]

    ds.ImagePositionPatient = list(header["ImagePositionPatient"])
    ds.SliceThickness = ""
    # Image Pixel
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = header["Rows"]
    ds.Columns = header["Columns"]
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 14
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.is_little_endian=True
    ds.is_implicit_VR=False

   # print("max pre post: {} {}".format(m1,m2))
    ds.PixelData = matrix.astype('<i2').tobytes() #make sure it's LE
    #ds.PixelData = matrix.tobytes()


    # MR Image
    ds.ImageType = 	'ORIGINAL/PRIMARY/OTHER'
    ds.ScanOptions = ""
    ds.ScanningSequence  = "RM"
    ds.SequenceVariant = "NONE"
    ds.RescaleIntercept = header["RescaleIntercept"]
    ds.RescaleSlope =  header["RescaleSlope"]
    ds.Manufacturer = "SIEMENS"


    # SOP Common
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'

    ds.SOPInstanceUID = SOPInstanceUID

    if bvalue is not None:
        block = ds.private_block(0x0019, "SIEMENS MR HEADER", create=True)

        block.add_new(0x0c, "IS", bvalue)



    ds.save_as(fileloc)

  #  nds=pydicom.read_file(fileloc)




def writeCTdicom(fileloc,matrix,header):
    # header info by IE (PS.3.3)
    SOPInstanceUID= pydicom.uid.generate_uid()
    file_meta = pydicom.dataset.Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    file_meta.MediaStorageSOPInstanceUID = SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    file_meta.FileMetaInformationGroupLength=10
    file_meta.FileMetaInformationGroupLength=file_meta.__sizeof__()
    ds = pydicom.dataset.FileDataset("dummy.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_implicit_VR = False
    # ds.is_little_endian = True
    # ds.is_implicit_VR = True

    # all are tpye 1 (mandatory) or type 2 (zero length if unknown otherwise actual value)
    # Patient

    ds.Modality = "CT"
    ds.PatientName = header["PatientName"]
    ds.PatientID =  header["PatientID"]
   # ds.PatientBirthDate = header["PatientBirthDate"]
    #ds.PatientSex = header["PatientSex"]
    # Study
    ds.StudyInstanceUID = header["StudyInstanceUID"]
    ds.SeriesDate = header["SeriesDate"]
    ds.SeriesTime = header["SeriesTime"]
    ds.StudyDate = header["StudyDate"]
    ds.StudyTime = header["StudyTime"]
    ds.ReferringPhysicianName = "Dr NonCon"
    ds.StudyID = ""
    ds.AccessionNumber = ""

    ds.StudyDescription =  header["StudyDescription"]
    # Series
    ds.SeriesDescription =  header["SeriesDescription"]

    ds.SeriesInstanceUID = header["SeriesInstanceUID"]
    ds.SeriesNumber = header["SeriesNumber"]
    ds.PatientPosition = "HFS"  # 2c but needed unless patient orientation code sequence is there

    # Frame of Reference
    ds.FrameOfReferenceUID =  header["FrameOfReferenceUID"]
    ds.PositionReferenceIndicator = ""
    # Equipment
    ds.Manufacturer = "StrokeCenter"

    # Image
    ds.InstanceNumber = header["InstanceNumber"]
    if "AcquisitionDate" in header:
        ds.AcquisitionDate = header["AcquisitionDate"]

    if "AcquisitionTime" in header:
        ds.AcquisitionTime =  header["AcquisitionTime"]

    # Image Plane
    ds.PixelSpacing =header["PixelSpacing"]
    ds.ImageOrientationPatient = header["ImageOrientationPatient"]

    ds.ImagePositionPatient = list(header["ImagePositionPatient"])
    ds.SliceThickness = ""
    # Image Pixel
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = header["Rows"]
    ds.Columns = header["Columns"]
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 16
    ds.PixelRepresentation = 1
    ds.is_little_endian=True
    ds.is_implicit_VR=True
    m1=matrix.max()
    #matrix=matrix.astype(int16)
    m2 = matrix.max()
   # print("max pre post: {} {}".format(m1,m2))
    ds.PixelData = matrix.astype('<i2').tobytes() #make sure it's LE
    #ds.PixelData = matrix.tobytes()


    # MR Image
    ds.ImageType = "ORIGINAL"
    ds.RescaleType = "HU"
    ds.RescaleIntercept = header["RescaleIntercept"]
    ds.RescaleSlope =  header["RescaleSlope"]
    ds.KVP = ""


    # SOP Common
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'

    ds.SOPInstanceUID = SOPInstanceUID

    ds.save_as(fileloc)

  #  nds=pydicom.read_file(fileloc)



def sitk2generic_mr(mr, nativeMRoutfolder, seriesheader=None,instanceNumber_start=0, bvalue = None):
    if not os.path.exists(nativeMRoutfolder):
        os.makedirs(nativeMRoutfolder)

    #defaults
    spacing=mr.img.GetSpacing()
    header={'SeriesInstanceUID':pydicom.uid.generate_uid(),
                    'StudyInstanceUID':pydicom.uid.generate_uid(),
                    'FrameOfReferenceUID':pydicom.uid.generate_uid(),
                    'ImageOrientationPatient': [1.0,0.0,0.0,0.0,1.0,0.0],
                    'ImagePositionPatient':[-100.0,-100.0,-100.0],    #startpoint, varying
                    'Rows':mr.arr.shape[1],
                    'Columns':mr.arr.shape[2],
                    'PixelSpacing':[ "{:2.8f}".format(spacing[1]),"{:2.8f}".format(spacing[0])],  #between rows / cols
                    'PatientID':'Noname',
                    'PatientName':'Noname',
                    'PatientBirthDate': '20010101',
                    'PatientSex': 'F',
                    'StudyDescription':'IndescriptStudy',
                    'SeriesDescription':'IndescriptSeries',
                    'MRAcquisitionType':'2D',
                    'RepetitionTime':'15000',
                    'EchoTime':'30',
                    'EchoTrainLength':'1',
                    'Modality':'MR',
                    'RescaleSlope':1,
                    'RescaleIntercept':0,
                    'SeriesTime':"120000",
                    'SeriesDate': "20000101",
                    'StudyTime': "120000",
                    'StudyDate': "20000101",
                    'SeriesNumber':'100'}


    if seriesheader:
        for key,val in seriesheader.items():
            header[key]=val

    IOP=np.array(header['ImageOrientationPatient'])

    nslices = mr.arr.shape[0]
    perp_vec = np.cross(IOP[0:3], IOP[3:6])
    interslicedistance =  spacing[2]
    origin=np.array(header['ImagePositionPatient'])

    for k in range(nslices):
        IPP = origin + k * perp_vec * interslicedistance
        #print(IPP)
        instancenumber = k+instanceNumber_start
        SOPInstanceUID = pydicom.uid.generate_uid()
        header["SOPInstanceUID"]=SOPInstanceUID
        header["InstanceNumber"] = instancenumber
        header["ImagePositionPatient"] = IPP
        if "AcquisitionTime" in header:
            header["AcquisitionTime"] = header["AcquisitionTime"]
        else:
            header["AcquisitionTime"] = header["SeriesTime"]

        if "AcquisitionDate" in header:
            header["AcquisitionDate"] = header["AcquisitionDate"]
        else:
            header["AcquisitionDate"] = header["StudyDate"]


        writeMRdicom(nativeMRoutfolder + '/' + SOPInstanceUID + '.dcm', mr.arr[k, :, :]-header['RescaleIntercept'],header,bvalue = bvalue)




def sitk2generic_ct(ct, ct_outfolder, seriesheader={},instance_number_start=0):
    if not os.path.exists(ct_outfolder):
        os.makedirs(ct_outfolder)

    ctarr = sitk.GetArrayViewFromImage(ct)
    spacing=ct.GetSpacing()
    header={'SeriesInstanceUID':pydicom.uid.generate_uid(),
            'StudyInstanceUID':pydicom.uid.generate_uid(),
            'FrameOfReferenceUID':pydicom.uid.generate_uid(),
            'Rows':ctarr.shape[1],
            'Columns':ctarr.shape[2],
            'PixelSpacing':[ "{:2.8f}".format(spacing[1]),"{:2.8f}".format(spacing[0])],  #between rows / cols
            'PatientID':'Noname',
            'PatientName':'Noname',
            'StudyDescription':'IndescriptStudy',
            'SeriesDescription':'IndescriptSeries',
            'Modality':'CT',
            'RescaleSlope':1,
            'RescaleIntercept':-1024,
            'SeriesNumber':'100'}

    header_use = header | seriesheader

    IOP=[f"{v:2.5}" for v in ct.GetDirection()[0:3]]

    nslices = ctarr.shape[0]

    for k in range(nslices):
        # convert to 3D single slice to get IPP
        IPP = ct[:,:,k:(k+1)].GetOrigin()

        instancenumber = k+instance_number_start
        SOPInstanceUID = pydicom.uid.generate_uid()
        header_use["SOPInstanceUID"] = SOPInstanceUID
        header_use["InstanceNumber"] = instancenumber
        header_use["ImagePositionPatient"] = IPP
        header_use["ImageOrientationPatient"] = IOP

        #right now assumes slope is 1
        writeCTdicom(ct_outfolder + '/' + SOPInstanceUID + '.dcm', ctarr[k, :, :]-header['RescaleIntercept'],header_use)





if __name__=="__main__":
    d = "/home/sorenc/DATA/AImedical/lukaku/incoming/coregProducts_A910789F-45AE-4589-803F-E01B6EBD52F3/002_BrainVsBrainNeck/fixImage"
    op = dicomscan(d)