import cv2
import numpy as np
import tensorflow as tf

def save_step_image(img, name):
    cv2.imwrite(f"./processing_steps/{name}.jpg", img)


class CharacterDetector:
    def __init__(self, loadFile: str):
        # Dictionary for getting characters from index values...
        self.word_dict = {}

        # Digit characters
        for i in range(0, 10):
            self.word_dict[i] = chr(48 + i)

        # Uppercase characters
        for i in range(10, 36):
            self.word_dict[i] = chr(
                65 - 10 + i
            )  # minus 10 because we have 10 digits already

        # Lowercase characters
        for i in range(36, 62):
            self.word_dict[i] = chr(
                97 - 10 - 26 + i
            )  # minus 10 because we have 10 digits already and minus 26 because we have 26 uppercase characters already

        if loadFile:
            self.model = tf.keras.models.load_model(loadFile)

    def predict(self, img):
        img_copy = img.copy()
        save_step_image(img_copy, "paint_window")

        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        save_step_image(img_gray, "grayscale")

        # Convert image such that background is black and foreground is white
        # as well as whereever there is color except white, make it black
        img_gray = np.where(img_gray < 255, 255, 0)
        img_gray = np.uint8(img_gray)
        save_step_image(img_gray, "white_black")

        # Detect bounding box and crop the image
        edged = cv2.Canny(img_gray, 10, 255)
        save_step_image(edged, "edged")

        contours, _ = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
        x, y, w, h = cv2.boundingRect(cnts[0])
        img_contour = img_gray.copy()
        cv2.drawContours(img_contour, cnts[0], -1, (0, 255, 0), 3)
        save_step_image(img_contour, "contours")

        img_crop = img_gray[y : y + h, x : x + w]
        img_final = cv2.resize(img_crop, (28, 28))

        img_final = img_final / 255.0
        save_step_image(img_final * 255, "model_input")

        img_final = np.reshape(img_final, (1, 28, 28, 1))
        prediction = self.model.predict(img_final)

        pred_prob = {}
        decimal_if_possible = list(map(float, prediction[0]))
        for idx, prob in enumerate(decimal_if_possible):
            pred_prob[self.word_dict[idx]] = prob

        print(
            {
                k: v
                for k, v in sorted(
                    pred_prob.items(), key=lambda item: item[1], reverse=True
                )
            }
        )

        img_pred = self.word_dict[np.argmax(prediction)]
        # cv2.putText(
        #     img,
        #     "Prediction: " + img_pred,
        #     (20, 410),
        #     cv2.FONT_HERSHEY_DUPLEX,
        #     1.3,
        #     color=(255, 0, 30),
        # )
        # cv2.imshow("Recognised Character", img)
        if __name__ == "__main__":
            while 1:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            cv2.destroyAllWindows()
        return img_pred
