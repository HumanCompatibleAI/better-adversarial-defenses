import cv2
from PIL import Image
from time import sleep
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.graphics.texture import Texture
import gym

env = gym.make('Reacher-v2')
env.reset()

kv = '''
BoxLayout:
    orientation: 'vertical'
    Label:
        text: app.obs
    BoxLayout:
        size_hint_y: None
        height: sp(100)
        orientation: 'vertical'
        BoxLayout:
            orientation: 'vertical'
            Slider:
                id: e1
                min: -10.
                max: 10.
                on_touch_move: app.log_slider(1, self.value)
            Label:
                text: 'a1 = {}'.format(e1.value)
        BoxLayout:
            orientation: 'vertical'
            Slider:
                id: e2
                min: -10.
                max: 10.
                on_touch_move: app.log_slider(2, self.value)
            Label:
                text: 'a2 = {}'.format(e2.value)

'''


class MuJoCoApp(App):
    obs = StringProperty("")
    def __init__(self, *args, **kwargs):
        super(MuJoCoApp, self).__init__(*args, **kwargs)
        self.sliders = [0, 0]
        self.idx = 0
    def log_slider(self, slider, a):
        self.sliders[slider - 1] = a
        obs, reward, done, info = env.step(self.sliders)
        self.obs = str(obs)
        img = env.render(mode='rgb_array')
        #img = Image.fromarray(img, 'RGB')
        #img.save('render%d.png' % self.idx)
        #self.image = 'render%d.png' % self.idx
        cv2.imshow("Image", img)
        self.idx += 1
        sleep(1)
        print(obs, reward, done, info)
        print(dir(self))
    def build(self):
        return Builder.load_string(kv)

MuJoCoApp().run()
cv2.destroyAllWindows()

