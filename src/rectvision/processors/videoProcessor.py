import numpy as np 
import cv2
import moviepy.editor
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
from .imageProcessor import ImageProcessor

class VideoProcessor():
    def __init__(self, video_path=None):
        if video_path is not None:
            self.video_path = video_path
            self.video = cv2.VideoCapture(video_path)
            self.imageProcessor = ImageProcessor()

    def videoFlipping(self, out_video_path, flip_dir='horizontal', video=None, video_path=None):
        '''
        This method flips every frame in the video in the direction specified by flip_dir
        and writes the resulting video to out_video_path.

        The parameters passed are:
        video: Video to flip.
        video_path: Path to video to flip.
        flip_dir: The direction to flip in ('horizontal' or 'vertical').
        out_video_path: The path to write the flipped video to. Can be .mp4 or .mov.

        '''
        if video is None:
            video = self.video
        if video_path is None:
            video_path = self.video_path

        new_frames = []
        flipped_frame_size = (0,0)
        fps = int(video.get(cv2.CAP_PROP_FPS))

        #extract audio from video
        self.mp4Tomp3('audio.mp3', video=VideoFileClip(video_path))
        

        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            flipped_frame = self.imageProcessor.imageFlipping(flip_dir=flip_dir, img=frame)
            new_frames.append(flipped_frame)
            flipped_frame_size = flipped_frame.shape[:2][::-1] #result as w, h

            #exit if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break
        
        #release videocapture object
        video.release

        #create videowriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('video.mp4', fourcc , fps, flipped_frame_size)
        #write newly flipped frames to video
        for i in range(len(new_frames)):
            out_video.write(new_frames[i])
        #release videowriter object    
        out_video.release()

        video = moviepy.editor.VideoFileClip('video.mp4')
        video.write_videofile(out_video_path, codec='mpeg4', audio='audio.mp3')
                
        print('SUCCESSFULLY FLIPPED VIDEO!')
        
        

    def mp4Tomp3(self, out_audio_path, video=None):
        '''
        This method converts an mp4 video to audio and writes the resulting audio to out_audio_path.

        The parameters passed are:
        video: Video to convert to mp3.
        out_audio_path: The path to write the mp3 audio to.

        '''
        if video is None:
            video = moviepy.editor.VideoFileClip(self.video_path)
        video.audio.write_audiofile(out_audio_path)
        print('SUCCESSFULLY CONVERTED VIDEO TO MP3!')

    def mp4ToMov(self, mov_video_path, video=None):
        '''
        This method converts an mp4 video to an mov video and writes the 
        resulting video to out_video_path.

        The parameters passed are:
        video: Video to convert to mov.
        out_video_path: The path to write the mov video to.

        '''
        if video is None:
            video = moviepy.editor.VideoFileClip(self.video_path)
        video.write_videofile(mov_video_path, codec='mpeg4')
        print('SUCCESSFULLY CONVERTED VIDEO TO MOV!')

    def mp4ToAvi(self, avi_video_path, video=None):
        '''
        This method converts an mp4 video to an avi video and writes the 
        resulting video to out_video_path.

        The parameters passed are:
        video: Video to convert to mov.
        out_video_path: The path to write the avi video to.

        '''
        if video is None:
            video = moviepy.editor.VideoFileClip(self.video_path)
        video.write_videofile(avi_video_path, codec='png')
        print('SUCCESSFULLY CONVERTED VIDEO TO AVI!')

    def insertSignature(self, signature_path, out_video_path, signature_start=(0,0), signature_size=None, video=None, video_path=None):
        '''
        This method inserts a signature to all frames of a video.

        The parameters passed are:
        signature_path: Path to the signature image to be inserted.
        out_video_path: Path to write signed video to. Can be .mp4 or .mov.
        signature_start: Tuple containing the starting coordinates of the label 
                         on the video frame in the form(h,w).
        signature_size: Tuple containing the size of the signature in the form (h,w)
        video: Video to insert signature.
        video_path: Path to video to insert signature.

        '''
        if video is None:
            video = self.video
        if video_path is None:
            video_path = self.video_path
        new_frames = []
        added_frame_size = (0,0)
        fps = int(video.get(cv2.CAP_PROP_FPS))

        #extract audio from video
        self.mp4Tomp3('audio.mp3', video=VideoFileClip(video_path))

        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            added_frame = self.imageProcessor.insertLabel(signature_path, label_start=signature_start, label_size=signature_size, img=frame)
            new_frames.append(added_frame)
            added_frame_size = added_frame.shape[:2][::-1] #result as w, h

            #exit if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break
        
        #release videocapture object
        video.release

        #create videowriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('video.mp4', fourcc , fps, added_frame_size)
        #write newly labelled frames to video
        for i in range(len(new_frames)):
            out_video.write(new_frames[i])
        #release videowriter object    
        out_video.release()

        #add audio to newly created video
        video = moviepy.editor.VideoFileClip('video.mp4')
        video.write_videofile(out_video_path, codec='mpeg4', audio='audio.mp3')

        print('SUCCESSFULLY INSERTED SIGNATURE TO VIDEO!')
        
    def insertLabel(self, out_video_path, label, label_color=(0,0,255), label_position=(20,20), font_scale=1.0, video=None, video_path=None):
        '''
        This method inserts a text label to all frames of a video.

        The parameters passed are:
        out_video_path: Path to write labelled video to. Can be .mp4 or .mov.
        label: Text to be inserted.
        label_color: Colour of text as tuple in the form (B, G, R) and values between 0 and 255.
        label_positioin: Tuple containing the coordinates of the bottom-left corner of label 
                         on the video frame in the form (x, y).
        font_scale: Scale to be multiplied by base-size of font type
        video: Video to insert text label.
        video_path: Path to video to insert text label.
        
        '''
        if video is None:
            video = self.video
        if video_path is None:
            video_path = self.video_path
        new_frames = []
        label_font = cv2.FONT_ITALIC
        labelled_frame_size = (0,0)
        fps = int(video.get(cv2.CAP_PROP_FPS))

        #extract audio from video
        self.mp4Tomp3('audio.mp3', video=VideoFileClip(video_path))

        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            labelled_frame = cv2.putText(img=frame, text=label, org=label_position, fontFace=label_font, fontScale=font_scale,color=label_color, thickness=5 )
            new_frames.append(labelled_frame)
            labelled_frame_size = labelled_frame.shape[:2][::-1] #result as w, h

            #exit if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break
        
        #release videocapture object
        video.release

        #create videowriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('video.mp4', fourcc , fps, labelled_frame_size)
        #write newly labelled frames to video
        for i in range(len(new_frames)):
            out_video.write(new_frames[i])
        #release videowriter object    
        out_video.release()

        #add audio to newly created video
        video = moviepy.editor.VideoFileClip('video.mp4')
        video.write_videofile(out_video_path, codec='mpeg4', audio='audio.mp3')

        print('SUCCESSFULLY INSERTED LABELS TO VIDEO!')
    
    def noiseCancellation(self):
        pass

    def trimVideo(self, start_trim, end_trim, trimmed_video_path, video_path=None):
        '''
        This method trims a video to span between start_trim and end_trim.

        The parameters passed are:
        start_trim: TIme mark to start the trim (in seconds).
        end_trim: Time mark to end the trim (in seconds).
        trimmed_video_path: Path to write trimmed video to.
        video_path: Path to video to be trimmed.
        
        '''
        if video_path is None:
            video_path = self.video_path
        ffmpeg_extract_subclip(filename=video_path, t1=start_trim, t2=end_trim, targetname=trimmed_video_path)
        print('SUCCESSFULLY TRIMED VIDEO!')

    def mergeVideo(self, add_vid_path, out_video_path, main_vid=None):
        '''
        This method merges two videos together.

        The parameters passed are:
        add_vid_path: Path to video to be added to a main video.
        out_video_path: Path to write merged video to.
        main_vid: Video to add another video to.

        '''
        if main_vid is None:
            main_vid = moviepy.editor.VideoFileClip(self.video_path)
        add_vid = moviepy.editor.VideoFileClip(add_vid_path)

        merged_vid = moviepy.editor.concatenate_videoclips([main_vid, add_vid])
        merged_vid.write_videofile(out_video_path, codec='libx264')
        print('SUCCESSFULLY MERGED VIDEOS!')