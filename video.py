from IPython.display import Video
import imageio


# show the video, from the list of rgb frames

# Assume 'frames' is a list of images (numpy arrays)
# Example: frames = [frame1, frame2, frame3, ...]

def create_video(frames, video_name='output_video.mp4'):

    # Create a video writer object using imageio
    with imageio.get_writer(video_name, fps=25) as writer:
        for frame in frames:
            # Convert the frame to an image and add it to the video
            writer.append_data(frame)

    # Display the video in the notebook
    Video(video_name, embed=True)
