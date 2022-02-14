from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

class CharRenderer:
    ''' A class to render single characters at correct height
    This is needed because Pillow doesn't have info about the character's y offset
    '''
    def __init__(self, font):
        self.font = font
        self.image = Image.new("RGBA", (300,100), (255,255,255,255))
        self.drawer = ImageDraw.Draw(self.image)
        self.fill = ' 0 '
        self.fill_width = font.getsize(self.fill)[0]

    def render(self, image, pos, character, color=(255,255,255)):
        full_width, full_height = self.font.getsize(self.fill + character)
        char_width = full_width - self.fill_width

        self.drawer.text((0,0), self.fill+character, fill=color, font=self.font)

        char_img = self.image.crop((self.fill_width,0, full_width,full_height))
        image.paste(char_img, pos, char_img)
        self.drawer.rectangle((0,0,300,100), (255,255,255,255))

font_fname = "explanation_resources/helvetica_bold.ttf"
font_size = 15
font = ImageFont.truetype(font_fname, font_size)
c = CharRenderer(font)
c.render(c.image, (50,0), '1')
c.image.show()