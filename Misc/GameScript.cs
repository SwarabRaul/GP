// 21BAI1225 - Swarab Raul

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class GameScript : MonoBehaviour
{
    public TMP_InputField boxVelocityInput;
    public Button startButton, dropButton;
    public TMP_Text resultText;
    public RectTransform box, ball;
    public float gravity = 9.81f;
    private float boxSpeed;
    private bool isBoxMoving, isBallFalling;

    // Start is called before the first frame update
    void Start()
    {
        startButton.onClick.AddListener(StartBoxMovement);
        dropButton.onClick.AddListener(DropBall);    
    }

    // Update is called once per frame
    void Update()
    {
        if(isBoxMoving) {
            MoveBox();
        }

        if(isBallFalling) {
            FallBall();
        }
    }

    public void StartBoxMovement() {
        if (float.TryParse(boxVelocityInput.text, out boxSpeed)) {
            isBoxMoving = true;
            resultText.text = "";
        } else {
            resultText.text = "Invalid Speed";
        }
    }

    public void DropBall() {
        isBallFalling = true;
    }

    void MoveBox() {
        box.anchoredPosition += new Vector2(boxSpeed * Time.deltaTime, 0);
    }

    void FallBall() {
        ball.anchoredPosition += new Vector2(0, -1 * gravity * Time.deltaTime);

        if(ball.anchoredPosition.y <= box.anchoredPosition.y) {  // Adjusted for box height check
            isBallFalling = false;
            CheckWinCondition();
        }
    }

    void CheckWinCondition() {
        if(Mathf.Abs(ball.anchoredPosition.x - box.anchoredPosition.x) < 100) {
            resultText.text = "You Win";
        } else {
            resultText.text = "Try Again";
        }
    }

    public void ResetGame() {
        box.anchoredPosition = new Vector2(-200, box.anchoredPosition.y);
        ball.anchoredPosition = new Vector2(0, 200);
        isBoxMoving = false;
        isBallFalling = false;
        resultText.text = "";
    }
}
